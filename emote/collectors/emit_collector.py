"""
Collector types for running environments
"""
from itertools import count
from typing import Dict, List

import numpy as np
import torch

from torch import nn

from emote.callbacks import LoggingCallback
from emote.proxies import AgentProxy, MemoryProxy
from emote.typing import AgentId, DictObservation, DictResponse, EpisodeState


class EmitWrapper:
    """Wraps a vectorised Emit isaac gym env.

    This currently converts the data back and forth between numpy torch. We should
    eventually convert this to keep the data on the GPU as pytorch tensors.

    :param env: (VecEnv) The vectorised gym env.
    """

    def __init__(self, venv, device, has_images: bool):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space

        self._next_agent = count()
        self._agent_ids: List[AgentId] = [
            next(self._next_agent) for i in range(self.num_envs)
        ]

        self._step_counter = np.zeros((self.num_envs,))
        self._has_images = has_images
        self._device = device

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def _copy_to_numpy(self, data):
        return data.clone().detach().cpu().numpy()

    def _copy_dict_to_numpy(self, data):
        return {key: self._copy_to_numpy(val) for (key, val) in data.items()}

    def _obs_to_array_data(self, converted_obs, env_id):
        array_data = {"obs": converted_obs["obs"][env_id][None][0]}
        if self._has_images:
            array_data["images"] = converted_obs["images"][env_id]
        return array_data

    def dict_step(
        self, actions: Dict[AgentId, DictResponse]
    ) -> Dict[AgentId, DictObservation]:

        batched_actions = torch.tensor(
            np.stack(
                [actions[agent].list_data["actions"] for agent in self._agent_ids]
            ),
            dtype=torch.float32,
            device=self._device,
        )
        next_obs, rewards, dones, infos = self.venv.step(batched_actions)

        new_agents = []
        results = {}

        rewards = self._copy_to_numpy(rewards)
        dones = self._copy_to_numpy(dones)
        converted_next_obs = self._copy_dict_to_numpy(next_obs)

        self._step_counter += 1

        for env_id in range(self.num_envs):
            completed_full_episode = dones[env_id] > 0
            # Break the episode into a series of smaller chunks in memory.
            # This makes training feasible when using isaac gym.
            # TODO: add memory logic that supports sampling from incomplete episodes
            # and then remove this split episode logic
            completed_split_episode = self._step_counter[env_id] > 100

            array_data = self._obs_to_array_data(converted_next_obs, env_id)

            if completed_full_episode or completed_split_episode:
                results[self._agent_ids[env_id]] = DictObservation(
                    episode_state=EpisodeState.TERMINAL
                    if completed_full_episode
                    else EpisodeState.INTERRUPTED,
                    array_data=array_data,
                    rewards={"reward": rewards[env_id]},
                )
                new_agent = next(self._next_agent)
                results[new_agent] = DictObservation(
                    episode_state=EpisodeState.INITIAL,
                    array_data=array_data,
                    rewards={"reward": None},
                )
                new_agents.append(new_agent)
                self._agent_ids[env_id] = new_agent
                self._step_counter[env_id] = 0

        for env_id, agent_id in enumerate(self._agent_ids):
            if agent_id not in new_agents:
                results.update(
                    {
                        agent_id: DictObservation(
                            episode_state=EpisodeState.RUNNING,
                            array_data=self._obs_to_array_data(
                                converted_next_obs, env_id
                            ),
                            rewards={"reward": rewards[env_id]},
                        )
                    }
                )

        ep_info = {}
        if "episode" in infos:
            for k, v in infos["episode"].items():
                ep_info[k] = self._copy_to_numpy(v)

        return results, ep_info

    def dict_reset(self) -> Dict[AgentId, DictObservation]:
        self._agent_ids = [next(self._next_agent) for i in range(self.num_envs)]
        self._step_counter = np.zeros((self.num_envs,))

        obs = self.venv.reset()
        converted_obs = self._copy_dict_to_numpy(obs)

        results = {}
        for env_id, agent_id in enumerate(self._agent_ids):
            results.update(
                {
                    agent_id: DictObservation(
                        episode_state=EpisodeState.INITIAL,
                        array_data=self._obs_to_array_data(converted_obs, env_id),
                        rewards={"reward": None},
                    )
                }
            )
        return results


class EmitAgentProxy:
    """This AgentProxy assumes that the observations will contain flat array of observations names 'obs'"""

    def __init__(self, policy: nn.Module, device: torch.device):
        self.policy = policy
        self._end_states = [EpisodeState.TERMINAL, EpisodeState.INTERRUPTED]
        self.device = device

    def __call__(
        self, observations: Dict[AgentId, DictObservation]
    ) -> Dict[AgentId, DictResponse]:
        """Runs the policy and returns the actions."""
        # The network takes observations of size batch x obs for each observation space.
        assert len(observations) > 0, "Observations must not be empty."
        active_agents = [
            agent_id
            for agent_id, obs in observations.items()
            if obs.episode_state not in self._end_states
        ]
        tensor_obs = torch.tensor(
            np.array(
                [observations[agent_id].array_data["obs"] for agent_id in active_agents]
            )
        ).to(self.device)
        actions = self.policy(tensor_obs)[0].clone().detach().cpu().numpy()
        return {
            agent_id: DictResponse(list_data={"actions": actions[i]}, scalar_data={})
            for i, agent_id in enumerate(active_agents)
        }


class EmitCollector(LoggingCallback):
    def __init__(
        self,
        env: EmitWrapper,
        agent: AgentProxy,
        memory: MemoryProxy,
        warmup_steps: int,
        inf_steps_per_bp: int,
    ):
        super().__init__()
        self._agent = agent
        self._env = env
        self._memory = memory
        self._warmup_steps = warmup_steps
        self._inf_steps_per_bp = inf_steps_per_bp

    def collect_data(self):
        actions = self._agent(self._obs)
        next_obs, ep_info = self._env.dict_step(actions)

        self._memory.add(self._obs, actions)
        self._obs = next_obs

        for k, v in ep_info.items():
            self.log_scalar("episode/" + k, v)

    def begin_training(self):
        self._obs = self._env.dict_reset()
        while self._memory.size() < self._warmup_steps:
            self.collect_data()

    def begin_batch(self):
        for _ in range(self._inf_steps_per_bp):
            self.collect_data()
