# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file contains codes and texts that are copied from
# https://github.com/facebookresearch/mbrl-lib

from collections import deque
from itertools import count
from typing import Optional, Union
from emote.callback import Callback
from emote.callbacks import LoggingMixin

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from emote.memory import MemoryLoader
from emote.models.model import DynamicModel
from emote.proxies import AgentProxy, MemoryProxy
from emote.typing import (
    AgentId,
    DictObservation,
    DictResponse,
    EpisodeState,
    RewardFnType,
    TensorType,
    TermFnType,
    BPStepScheduler,
)
from emote.utils.math import truncated_linear
from emote.utils.model import to_numpy, to_tensor


class ModelEnv:
    """Wraps a dynamics model into a gym-like environment.
    Args:
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
        self._current_obs: torch.Tensor = None
        self._init_obs: torch.Tensor = None
        self._propagation_method: Optional[str] = None
        self._model_indices = None
        self._timestep = 0
        self._len_rollout = 0
        if generator:
            self.rng = generator
        else:
            self.rng = torch.Generator(device=self.device)
        self._next_agent = count()
        self._agent_ids: list[AgentId] = [
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
        self._current_obs = to_tensor(initial_obs_batch).to(self.dynamics_model.device)
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
            (Union[tuple, dict]): contains the predicted next observation, reward, done flag.
            The done flag and rewards are computed using the termination_fn and
            reward_fn passed in the constructor. The rewards can also be predicted
            by the model.
        """
        assert len(actions.shape) == 2  # batch, action_dim
        with torch.no_grad():
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).to(self.device)
            (next_observs, pred_rewards,) = self.dynamics_model.sample(
                action=actions,
                observation=self._current_obs,
                rng=self.rng,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            dones = self.termination_fn(actions, next_observs)

            info = {"terminated": torch.zeros(dones.shape)}
            self._timestep += 1
            if self._timestep >= self._len_rollout:
                info["terminated"] += 1.0
            self._current_obs = torch.clone(next_observs)
            return next_observs, rewards, dones, info

    def dict_step(
            self, actions: dict[AgentId, DictResponse]
    ) -> tuple[dict[AgentId, DictObservation], dict[str, float]]:
        batched_actions = np.stack(
            [actions[agent].list_data["actions"] for agent in self._agent_ids]
        )
        next_obs, rewards, dones, info = self.step(batched_actions)
        new_agents = []
        results = {}
        terminated = info["terminated"]

        for env_id, (done, term) in enumerate(zip(dones, terminated)):
            if done or term:
                episode_state = (
                    EpisodeState.TERMINAL if done else EpisodeState.INTERRUPTED
                )
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

    def dict_reset(
            self,
            obs: TensorType,
            len_rollout: int,
    ) -> dict[AgentId, DictObservation]:
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


class BatchCallback(LoggingMixin, Callback):
    def __init__(self):
        super().__init__()

    def begin_batch(self, *args, **kwargs):
        pass

    @Callback.extend
    def collect_multiple(self, *args, **kwargs):
        pass

    @Callback.extend
    def clone_batch(self, *args, **kwargs):
        pass


class BatchSampler(BatchCallback):
    def __init__(
            self,
            dataloader: MemoryLoader,
            model_data_prob_schedule: BPStepScheduler,
            data_group: str = "rl_buffer",
            generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.dataloader = dataloader
        self.data_group = data_group
        self.iter = iter(self.dataloader)
        self.scheduler = model_data_prob_schedule
        self.prob_of_sampling_model_data = self.scheduler.value_min
        self.rng = generator if generator else torch.Generator()
        self.bp_counter = 0

    def begin_batch(self, *args, **kwargs):
        """ Generates a batch of data either by sampling from the model buffer or by
            cloning the input batch
            Returns:
                (dict): the batch of data
        """
        self.log_scalar(
            "training/prob_sampling_from_model", self.prob_of_sampling_model_data
        )
        if self.use_model_batch():
            batch = self.sample_model_batch()
        else:
            batch = self.clone_batch(*args, **kwargs)
        return {self.data_group: batch}

    def clone_batch(self,
                    observation,
                    actions,
                    next_observation,
                    rewards,
                    masks):
        """Clone the input batch
        """
        return {'observation': observation,
                'actions': actions,
                'next_observation': next_observation,
                'rewards': rewards,
                'masks': masks}

    def sample_model_batch(self):
        """Samples a batch of data from the model buffer
            Returns:
                (dict): batch samples
        """
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            batch = next(self.iter)
        return batch

    def use_model_batch(self):
        """Decides if batch should come from the model-generated buffer
            Returns:
                (bool): True if model samples should be used, False otherwise.
        """
        self.bp_counter += 1
        self.prob_of_sampling_model_data = truncated_linear(
            min_x=self.scheduler.bp_step_begin,
            max_x=self.scheduler.bp_step_end,
            min_y=self.scheduler.value_min,
            max_y=self.scheduler.value_max,
            x=self.bp_counter
        )
        rnd = torch.rand(size=(1,), generator=self.rng)[0]
        return True if rnd < self.prob_of_sampling_model_data else False


class ModelBasedCollector(BatchCallback):
    def __init__(
            self,
            model_env: ModelEnv,
            agent: AgentProxy,
            memory: MemoryProxy,
            rollout_scheduler: BPStepScheduler,
            num_bp_to_retain_buffer=1000000,
    ):
        super().__init__()
        self.agent = agent
        self.memory = memory
        self.model_env = model_env
        self.last_environment_rewards = deque(maxlen=1000)

        self.len_rollout = int(rollout_scheduler.value_min)
        self.rollout_scheduler = rollout_scheduler
        self.num_bp_to_retain_buffer = num_bp_to_retain_buffer
        self.obs: dict[AgentId, DictObservation] = None
        self.prob_of_sampling_model_data = 0.0
        self.bp_counter = 0

    def begin_batch(self, *args, **kwargs):
        self.update_rollout_size()
        self.log_scalar("training/model_rollout_length", self.len_rollout)
        self.collect_multiple(*args, **kwargs)

    def collect_multiple(self, observation):
        """Collect multiple rollouts
        :param observation: initial observations
        """
        self.obs = self.model_env.dict_reset(observation["obs"], self.len_rollout)
        for _ in range(self.len_rollout + 1):
            self.collect_sample()

    def collect_sample(self):
        """Collect a single rollout"""
        actions = self.agent(self.obs)
        next_obs, ep_info = self.model_env.dict_step(actions)

        self.memory.add(self.obs, actions)
        self.obs = next_obs

        if "reward" in ep_info:
            self.log_scalar("episode/model_reward", ep_info["reward"])

    def update_rollout_size(self):
        self.bp_counter += 1
        len_rollout = int(
            truncated_linear(min_x=self.rollout_scheduler.bp_step_begin,
                             max_x=self.rollout_scheduler.bp_step_end,
                             min_y=self.rollout_scheduler.value_min,
                             max_y=self.rollout_scheduler.value_max,
                             x=self.bp_counter,
                             )
        )
        if self.len_rollout != len_rollout:
            self.len_rollout = len_rollout
            new_memory_size = (
                    self.len_rollout
                    * self.model_env.num_envs
                    * self.num_bp_to_retain_buffer
            )
            self.memory.resize(new_memory_size)
