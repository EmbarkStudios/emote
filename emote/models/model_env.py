# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file contains codes/text mostly restructured from the following github repository
# https://github.com/facebookresearch/mbrl-lib

import gym
import numpy as np
import torch
from itertools import count
from typing import Dict, List, Optional, Tuple, Callable, Union

from torch import Tensor

from emote.models.model import DynamicModel
from emote.typing import TensorType, AgentId, DictObservation, \
                         DictResponse, EpisodeState, RewardFnType, TermFnType
from emote.utils.model import to_tensor, to_numpy
from tests.gym.collector import CollectorCallback
from emote.proxies import AgentProxy, MemoryProxy
from emote.memory import MemoryLoader
from emote.utils.math import truncated_linear
from collections import deque


class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.

    Args:
        env (gym.Env): the original gym environment for which the model was trained.
        num_envs (int): the number of envs to simulate in parallel (batch_size).
        model (DynamicModel): the dynamic model to wrap.
        termination_fn (callable): a function that receives actions and observations, and
            returns a boolean flag indicating whether the episode should end or not.
        reward_fn (callable, optional): a function that receives actions and observations
            and returns the value of the resulting reward in the environment.
        generator (torch.Generator, optional): a torch random number generator
    """

    def __init__(
            self,
            env: gym.Env,
            num_envs: int,
            model: DynamicModel,
            termination_fn: TermFnType,
            reward_fn: Optional[RewardFnType] = None,
            generator: Optional[torch.Generator] = None,
    ):
        self.dynamics_model = model
        self.termination_fn = termination_fn
        self.reward_fn = reward_fn
        self.device = model.device
        self.num_envs = num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._current_obs: torch.Tensor = None
        self._init_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        self._timestep = 0
        self._len_rollout = 0
        if generator:
            self._rng = generator
        else:
            self._rng = torch.Generator(device=self.device)
        self._next_agent = count()
        self._agent_ids: List[AgentId] = [
            next(self._next_agent) for i in range(self.num_envs)
        ]

    def reset(
            self,
            initial_obs_batch: TensorType,
            len_rollout: int,
    ):
        """Resets the model environment.

        Args:
            initial_obs_batch (TensorType): a batch of initial observations.
            len_rollout (int): the max length of the model rollout
        """
        self._timestep = 0
        self._len_rollout = len_rollout

        assert len(initial_obs_batch.shape) == 2  # batch, obs_dim
        self._current_obs = to_tensor(
            initial_obs_batch
        ).to(self.dynamics_model.device)
        self._init_obs = torch.clone(self._current_obs)

    def step(
            self,
            actions: TensorType,
    ) -> tuple[Tensor, Tensor | None, Tensor, dict[str, Tensor]]:
        """Steps the model environment with the given batch of actions.
        Args:
            actions (torch.Tensor or np.ndarray): the actions for each "episode" to rollout.
                Shape must be batch_size x dim_actions. If a np.ndarray is given, it's
                converted to a torch.Tensor and sent to the model device.

        Returns:
            (Union[tuple, Dict]): contains the predicted next observation, reward, done flag.
            The done flag and rewards are computed using the termination_fn and
            reward_fn passed in the constructor. The rewards can also be predicted
            by the model.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            (
                next_observs,
                pred_rewards,
            ) = self.dynamics_model.sample(
                action=actions,
                observation=self._current_obs,
                rng=self._rng,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            dones = self.termination_fn(actions, next_observs)

            info = {'terminated': torch.zeros(dones.shape)}
            self._timestep += 1
            if self._timestep >= self._len_rollout:
                info['terminated'] += 1.0
            self._current_obs = torch.clone(next_observs)
            return next_observs, rewards, dones, info

    def dict_step(
            self, actions: Dict[AgentId, DictResponse]
    ) -> Tuple[Dict[AgentId, DictObservation], Dict[str, float]]:
        batched_actions = np.stack(
            [actions[agent].list_data["actions"] for agent in self._agent_ids]
        )
        next_obs, rewards, dones, info = self.step(batched_actions)
        new_agents = []
        results = {}
        terminated = info['terminated']

        for env_id, (done, term) in enumerate(zip(dones, terminated)):
            if done or term:
                episode_state = EpisodeState.TERMINAL if done else EpisodeState.INTERRUPTED
                results[self._agent_ids[env_id]] = DictObservation(
                    episode_state=episode_state,
                    array_data={"obs": to_numpy(next_obs[env_id])},
                    rewards={"reward": to_numpy(rewards[env_id])},
                )
                new_agent = next(self._next_agent)
                results[new_agent] = DictObservation(
                    episode_state=EpisodeState.INITIAL,
                    array_data={"obs": to_numpy(self._init_obs[env_id])},
                    rewards={"reward": None},
                )
                new_agents.append(new_agent)
                self._agent_ids[env_id] = new_agent
        results.update(
            {
                agent_id: DictObservation(
                    episode_state=EpisodeState.RUNNING,
                    array_data={"obs": to_numpy(next_obs[env_id])},
                    rewards={"reward": to_numpy(rewards[env_id])},
                )
                for env_id, agent_id in enumerate(self._agent_ids)
                if agent_id not in new_agents
            }
        )
        ep_info = {}
        return results, ep_info

    def dict_reset(self,
                   obs: TensorType,
                   len_rollout: int,
                   ) -> Dict[AgentId, DictObservation]:
        """resets the model env.
        Args:
            obs (torch.Tensor or np.ndarray): the initial observations.
            len_rollout (int): the max rollout length

        Returns:
            (dict): the formatted initial observation.
        """
        self.reset(obs, len_rollout)
        self._agent_ids = [next(self._next_agent) for _ in range(self.num_envs)]
        return {
            agent_id: DictObservation(
                episode_state=EpisodeState.INITIAL,
                array_data={"obs": to_numpy(obs[i])},
                rewards={"reward": None},
            )
            for i, agent_id in enumerate(self._agent_ids)
        }


class ModelBasedCollector(CollectorCallback):
    def __init__(
            self,
            model_env: ModelEnv,
            agent: AgentProxy,
            memory: MemoryProxy,
            dataloader: MemoryLoader,
            rollout_schedule: List[int] = [0, 1000, 4, 4],  # [bp_min, bp_max, length_min, length_max]
            data_group_prob_schedule: List[float] = [0, 1000, 0.1, 0.95],  # [bp_min, bp_max, prob_min, prob_max]
            num_bp_to_retain_buffer = 1000000,
            data_group: str = "default",
            generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.data_group = data_group
        self.data_group_locked = True
        self._data_group_to_dictate = 'default'
        self._agent = agent
        self._memory = memory
        self._dataloader = dataloader
        self._iter = iter(self._dataloader)
        self._model_env = model_env
        self._last_environment_rewards = deque(maxlen=1000)

        self._len_rollout = rollout_schedule[2]
        self._rollout_schedule = rollout_schedule
        self._prob_of_sampling_model_data = data_group_prob_schedule[2]
        self._data_group_prob_schedule = data_group_prob_schedule
        self.num_bp_to_retain_buffer = num_bp_to_retain_buffer
        self._obs: Dict[AgentId, DictObservation] = None
        self._prob_of_sampling_model_data = 0.0
        self._rng = generator if generator else torch.Generator()
        self.bp_counter = 0

    def begin_batch(self, *args, **kwargs):
        self.bp_counter += 1
        self.update_rollout_size()
        self.update_data_group()

        self.log_scalar("training/prob_sampling_from_model", self._prob_of_sampling_model_data)
        self.log_scalar("training/model_rollout_length", self._len_rollout)

        mb_data_group = True \
            if self._data_group_to_dictate == 'model_samples' \
            else False

        self.log_scalar("training/mb_data_group", mb_data_group)

        self.collect_multiple(*args, **kwargs)
        if self._data_group_to_dictate == "model_samples" and \
                self._memory.size() > 1000:
            try:
                batch = next(self._iter)
            except StopIteration:
                self._iter = iter(self._dataloader)
                batch = next(self._iter)
            return {"model_samples": batch,
                    "data_group": "model_samples"}
        return {"data_group": "default"}

    def collect_multiple(self, observation):
        """Collect multiple rollouts

        :param observation: initial observations
        """
        self._obs = self._model_env.dict_reset(observation['obs'], self._len_rollout)
        for _ in range(self._len_rollout + 1):
            self.collect_sample()

    def collect_sample(self):
        """Collect a single rollout"""
        actions = self._agent(self._obs)
        next_obs, ep_info = self._model_env.dict_step(actions)

        self._memory.add(self._obs, actions)
        self._obs = next_obs

        if "reward" in ep_info:
            self.log_scalar("episode/model_reward", ep_info["reward"])

    def update_data_group(self):
        self._prob_of_sampling_model_data = truncated_linear(
            *(self._data_group_prob_schedule + [self.bp_counter + 1])
        )
        rnd = torch.rand(size=(1,), generator=self._rng)[0]
        self._data_group_to_dictate = "model_samples" \
            if rnd < self._prob_of_sampling_model_data \
            else "default"

    def update_rollout_size(self):
        len_rollout = int(
            truncated_linear(
                *(self._rollout_schedule + [self.bp_counter + 1])
            )
        )
        if self._len_rollout != len_rollout:
            self._len_rollout = len_rollout
            new_memory_size = (
                    self._len_rollout *
                    self._model_env.num_envs *
                    self.num_bp_to_retain_buffer
            )
            self._memory.resize(new_memory_size)

