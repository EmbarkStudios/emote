# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file contains codes and texts that are copied from
# https://github.com/facebookresearch/mbrl-lib

from itertools import count
from typing import Optional, Union

import numpy as np
import torch

from torch import Tensor

from emote.models.model import DynamicModel
from emote.typing import (
    AgentId,
    DictObservation,
    DictResponse,
    EpisodeState,
    RewardFnType,
    TensorType,
    TermFnType,
)
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
        self.dynamic_model = model
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
        self._current_obs = to_tensor(initial_obs_batch).to(self.dynamic_model.device)
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
            (next_observs, pred_rewards,) = self.dynamic_model.sample(
                action=actions,
                observation=self._current_obs,
                rng=self.rng,
            )
            rewards = (
                pred_rewards
                if self.reward_fn is None
                else self.reward_fn(actions, next_observs)
            )
            dones = self.termination_fn(next_observs)

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
