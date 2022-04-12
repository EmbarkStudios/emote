"""
Collector types for running environments
"""
from collections import deque, defaultdict
import collections
from itertools import count
import threading
from typing import Dict, List
import time
import torch
import numpy as np
import gym.spaces as gspaces
from torch import nn

from emote.callback import Callback
from emote.callbacks import LoggingCallback
from emote.typing import (
    DictResponse,
    EpisodeState,
    DictObservation,
    AgentId,
    MetaData,
)
from emote.proxies import AgentProxy, MemoryProxy


class EmitWrapper:
    """Wraps a vectorised isaac gym env to make it compatible with the hive workflow.

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

        self._last_environment_rewards = deque(maxlen=1000)
        self._step_counter = np.zeros((self.num_envs,))
        self._has_images = has_images
        self._episode_rewards: List[float] = [0.0 for i in range(self.num_envs)]
        self._device = device

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def _copy_to_numpy(self, data):
        return data.detach().clone().cpu().numpy()

    def _copy_obs_to_numpy(self, data):
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
        next_obs, rewards, dones, info = self.venv.step(batched_actions)

        info["timeouts"] = info["time_outs"].cpu().numpy()
        info["num_steps"] = info["num_steps"].cpu().numpy()

        new_agents = []
        results = {}
        completed_episode_rewards = []

        # actions = {key: self._copy_to_numpy(val) for (key, val) in actions.items()}
        rewards = self._copy_to_numpy(rewards)
        dones = self._copy_to_numpy(dones)
        converted_next_obs = self._copy_obs_to_numpy(next_obs)

        for env_id in range(self.num_envs):
            self._episode_rewards[env_id] += rewards[env_id]

        self._step_counter += 1

        for env_id in range(self.num_envs):
            completed_full_episode = dones[env_id] > 0
            # Break the episode into a series of smaller chunks in memory.
            # This makes training feasible when using isaac gym.
            # Note: This isn't ideal, since currently we lose the final reward.
            # of split episodes. TODO: fix this.
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
                completed_episode_rewards.append(self._episode_rewards[env_id])
                self._agent_ids[env_id] = new_agent
                self._episode_rewards[env_id] = 0.0
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
        if len(completed_episode_rewards) > 0:
            ep_info["reward"] = sum(completed_episode_rewards) / len(
                completed_episode_rewards
            )

        return results, ep_info

    def dict_reset(self) -> Dict[AgentId, DictObservation]:
        self._agent_ids = [next(self._next_agent) for i in range(self.num_envs)]
        self._step_counter = np.zeros((self.num_envs,))

        obs = self.venv.reset()
        converted_obs = self._copy_obs_to_numpy(obs)

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

    def close(self):
        pass

    def render(self):
        pass

    def step_wait(self):
        pass


class EmitCollector(LoggingCallback):
    def __init__(
        self,
        env: EmitWrapper,
        agent: AgentProxy,
        memory: MemoryProxy,
    ):
        super().__init__()
        self._agent = agent
        self._env = env
        self._memory = memory
        self._stop = False
        self._thread = None

    def collect_forever(self):
        self._obs = self._env.dict_reset()
        while not self._stop:
            self.collect_data()

    def collect_data(self):
        actions = self._agent(self._obs)
        next_obs, ep_info = self._env.dict_step(actions)

        self._memory.add(self._obs, actions)
        self._obs = next_obs

        if "reward" in ep_info:
            self.log_scalar("episode/reward", ep_info["reward"])

    def begin_training(self):
        self._thread = threading.Thread(target=self.collect_forever)
        self._thread.start()

    def end_training(self):
        self._stop = True

        if self._thread is not None:
            self._thread.join()
            self._thread = None
