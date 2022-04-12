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
from gym.vector import VectorEnvWrapper, VectorEnv
from torch import nn

from emote.callback import Callback
from emote.typing import (
    DictResponse,
    EpisodeState,
    DictObservation,
    AgentId,
    HiveResponse,
    MetaData,
)
from emote.proxies import AgentProxy, MemoryProxy


class HiveEmitWrapper:

    """Wraps a vectorised gym env to make it compatible with the hive workflow.

    :param env: (VecEnv) The vectorised gym env.
    """

    def __init__(self, env):
        self.venv = env
        if isinstance(self.venv.observation_space, gspaces.Tuple):
            raise NotImplementedError(
                "Tuple observations spaces are not currently supported."
            )

    def _create_array_data_from_state(self, state, idx):
        array_data = {}
        for k, v in state.items():
            array_data[k] = v[idx]
        return array_data

    def _create_numpy_state_dict(self, state_dict):
        # state_dict['features'] = state_dict.pop("obs")
        state = {k: v.cpu().numpy() for k, v in state_dict.items()}
        return state

    def reset(self, **kwargs):
        state_dict = self.venv.reset()
        state = self._create_numpy_state_dict(state_dict)

        obs_dict = {}
        self._cumulative_reward = {}
        self._episode_lengths = {}

        # Use the index as the initial agent id.
        for idx in range(self.venv.num_envs):
            obs_dict[idx] = DictObservation(
                array_data=self._create_array_data_from_state(state, idx),
                rewards={"reward": 0.0},
                episode_state=EpisodeState.INITIAL,
                metadata=MetaData(),
            )

            self._cumulative_reward[idx] = 0
            self._episode_lengths[idx] = 0

        return obs_dict

    def step(self, actions_dict):
        agent_ids, agent_actions = zip(*actions_dict.items())
        agent_actions = [_act.list_data["actions"] for _act in agent_actions]
        agent_actions = torch.tensor(agent_actions, device=self.venv.device)

        state_dict, rewards, dones, infos = self.venv.step(agent_actions)
        state = self._create_numpy_state_dict(state_dict)

        obs_dict = {}
        rewards_dict = {}
        dones_dict = {}
        for cc, idx in enumerate(agent_ids):
            rewards_dict[idx] = rewards[cc]
            dones_dict[idx] = dones[cc]
            done = bool(dones[cc])

            info = {
                "instant/reward": rewards[cc],
                "instant/time-alive": self._episode_lengths[idx],
            }

            self._cumulative_reward[idx] += rewards[cc]
            self._episode_lengths[idx] += 1

            if done:
                info["episode/reward"] = self._cumulative_reward.pop(idx)
                info["episode/length"] = self._episode_lengths.pop(idx)

            obs = DictObservation(
                array_data=self._create_array_data_from_state(state, cc),
                rewards={"reward": rewards[cc]},
                episode_state=EpisodeState.TERMINAL if done else EpisodeState.RUNNING,
                metadata=MetaData(info=info),
            )
            obs_dict[idx] = obs

            if done:
                # Create a unique id for the new episode.
                new_id = idx + self.venv.num_envs
                assert new_id not in obs_dict
                new_obs = DictObservation(
                    array_data=self._create_array_data_from_state(state, cc),
                    rewards={"reward": 0.0},
                    episode_state=EpisodeState.INITIAL,
                    metadata=MetaData(
                        info={
                            "instant/reward": 0.0,
                            "instant/time-alive": 0,
                        }
                    ),
                )

                obs_dict[new_id] = new_obs
                dones_dict[new_id] = new_obs.done
                rewards_dict[new_id] = new_obs.reward
                self._cumulative_reward[new_id] = 0
                self._episode_lengths[new_id] = 0

        return obs_dict, rewards_dict, dones_dict

    def close(self):
        self.venv.close()

    def step_wait(self):
        return self.venv.step_wait()

    def render(self):
        pass


class EmitCollector(Callback):
    MAX_NUMBER_REWARDS = 1000

    def __init__(
        self,
        env: HiveEmitWrapper,
        agent: AgentProxy,
        memory: MemoryProxy,
        render: bool = True,
    ):
        super().__init__()
        self._stop = False
        self._thread = None
        self._agent = agent
        self._memory = memory
        self._env = env
        self._render = render
        self._reward_buffer = collections.deque(maxlen=10)

        self.num_envs = 1
        self._total_inf_steps: int = 1_000_000_000
        self._time_limit: int = 36_000
        self._log: bool = False
        self._log_interval: int = 1000

    def collect_data(self):
        """
        Warning: This game loop is very slow, due to the necessity of splitting the inf batch.
        """
        episode_rewards = defaultdict(float)

        obs_dict = self._env.reset()

        start = time.perf_counter()
        update_step = 0
        for inf_step in range(0, self._total_inf_steps, self.num_envs):
            if (
                self._log
                and (inf_step % self._log_interval == 0)
                and (len(self._reward_buffer) > 0)
            ):
                print(
                    f"Step: {inf_step}, Mean Reward: {sum(self._reward_buffer)/len(self._reward_buffer)}"
                )

            actions_dict = self._agent(obs_dict)
            self._memory.add(obs_dict, actions_dict)
            obs_dict, _, done_dict = self._env.step(actions_dict)

            for idx in obs_dict.keys():
                episode_rewards[idx] += obs_dict[idx].rewards["reward"]
                if done_dict[idx]:
                    self._reward_buffer.append(episode_rewards.pop(idx))

            if self._render:
                self._env.render()

            update_step += 1
            elapsed = time.perf_counter() - start
            if elapsed > self._time_limit:
                break

        # self._env.close()

    def collect_forever(self):
        while not self._stop:
            self.collect_data()

    def begin_training(self):
        self._thread = threading.Thread(target=self.collect_forever)
        self._thread.start()

    def end_training(self):
        self._stop = True

        if self._thread is not None:
            self._thread.join()
            self._thread = None
            # self._env.close()


class EmitWrapper2:
    """Wraps a vectorised isaac gym env to make it compatible with the hive workflow.

    :param env: (VecEnv) The vectorised gym env.
    """

    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space

    def reset(self, **kwargs):
        return self.venv.reset(**kwargs)

    def step(self, agent_actions):
        state_dict, rewards, dones, infos = self.venv.step(agent_actions)

        infos["timeouts"] = infos["time_outs"].cpu().numpy()
        infos["num_steps"] = infos["num_steps"].cpu().numpy()

        return state_dict, rewards, dones, infos

    def close(self):
        pass

    def render(self):
        pass

    def step_wait(self):
        pass


class EmitFeatureAgentProxy:
    """This AgentProxy assumes that the observations will contain flat array of observations names 'obs'"""

    def __init__(self, policy: nn.Module):
        self.policy = policy
        self._end_states = [EpisodeState.TERMINAL, EpisodeState.INTERRUPTED]

    def __call__(self, observations: Dict[str, torch.tensor]):
        tensor_obs = observations["obs"].detach()
        actions = self.policy(tensor_obs)[0]
        return actions.detach()


class EmitCollector2(Callback):
    MAX_NUMBER_REWARDS = 1000

    def __init__(
        self,
        env,
        agent: AgentProxy,
        memory: MemoryProxy,
        render: bool = True,
        has_images: bool = False,
    ):
        super().__init__()
        self._agent_proxy = agent
        self._env = EmitWrapper2(env)
        self._memory = memory

        self._next_agent = count()
        self._agent_ids: List[AgentId] = [
            next(self._next_agent) for i in range(self.num_envs)
        ]

        self._env = env
        self._num_envs = env.num_envs
        self._last_environment_rewards = deque(maxlen=1000)
        self._stop = False
        self._thread = None
        self._step_counter = np.zeros((self._num_envs,))
        self._has_images = has_images

        self._episode_rewards: List[float] = [0.0 for i in range(self._num_envs)]

    def _convert_data(self, data):
        return data.detach().clone().cpu().numpy()

    def _convert_obs_data(self, data):
        return {key: self._convert_data(val) for (key, val) in data.items()}

    def dict_step(
        self, actions: Dict[AgentId, DictResponse]
    ) -> Dict[AgentId, DictObservation]:

        next_obs, rewards, dones, info = self._env.step(actions)
        new_agents = []
        results = {}
        completed_episode_rewards = []

        actions = self._convert_data(actions)
        rewards = self._convert_data(rewards)
        dones = self._convert_data(dones)
        converted_next_obs = self._convert_obs_data(next_obs)

        for env_id in range(self._num_envs):
            self._episode_rewards[env_id] += rewards[agent_id]

        self._step_counter += 1

        for env_id in range(self._num_envs):
            completed_full_episode = dones[env_id] > 0
            # Break the episode into a series of smaller chunks in memory.
            # This makes training feasible when using isaac gym.
            # Note: This isn't ideal, since currently we lose the final reward.
            # of split episodes. TODO: fix this.
            completed_split_episode = self._step_counter[env_id] > 100

            array_data = {"obs": converted_next_obs["obs"][env_id][None][0]}
            if self._has_images:
                array_data["images"] = converted_next_obs["images"][env_id]

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
                array_data = {"obs": converted_next_obs["obs"][env_id][None][0]}
                if self._has_images:
                    array_data["images"] = converted_next_obs["images"][env_id]
                results.update(
                    {
                        agent_id: DictObservation(
                            episode_state=EpisodeState.RUNNING,
                            array_data=array_data,
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

    def collect_forever(self):
        self._obs = self.dict_reset()

        while not self._stop:
            actions = self._agent_proxy(self._obs)
            next_obs, ep_info = self.dict_step(actions)

            self._memory.add(self._obs, actions)
            self._obs = next_obs

            if "reward" in ep_info:
                self.log_scalar("episode/reward", ep_info["reward"])

    def dict_reset(self) -> Dict[AgentId, DictObservation]:
        self._agent_ids = [next(self._next_agent) for i in range(self._num_envs)]
        self._step_counter = np.zeros((self._num_envs,))

        obs = self._env.reset()
        converted_obs = self._convert_obs_data(obs)

        results = {}
        for env_id, agent_id in enumerate(self._agent_ids):
            array_data = {"obs": converted_obs["obs"][env_id][None][0]}
            if self._has_images:
                array_data["images"] = converted_obs["images"][env_id]
            results.update(
                {
                    agent_id: DictObservation(
                        episode_state=EpisodeState.INITIAL,
                        array_data=array_data,
                        rewards={"reward": None},
                    )
                }
            )
        return results

    def begin_training(self):
        self._thread = threading.Thread(target=self.collect_forever)
        self._thread.start()

    def end_training(self):
        self._stop = True

        if self._thread is not None:
            self._thread.join()
            self._thread = None
