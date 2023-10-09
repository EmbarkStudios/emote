from __future__ import annotations
import copy

from typing import Optional
import numpy as np

import torch
from torch import nn, optim
from emote.callback import Callback
from emote.callbacks.loss import LossCallback
from emote.mixins.logging import LoggingMixin
from emote.proxies import AgentProxy
from emote.algorithms.sac import soft_update_from_to
from emote.typing import AgentId, DictObservation, DictResponse, EpisodeState
from emote.utils.gamma_matrix import discount, make_gamma_matrix, split_rollouts
from emote.utils.spaces import MDPSpace


class GenericAgentProxy(AgentProxy):
    """Observations are dicts that contain multiple input and output keys.

    For example, we might have a policy that takes in both "obs" and
    "goal" and outputs "actions". In order to be able to properly
    invoke the network it is the responsibility of this proxy to
    collate the inputs and decollate the outputs per agent.
    """

    def __init__(
        self,
        policy: nn.Module,
        device: torch.device,
        input_keys: tuple,
        output_keys: tuple,
        spaces: MDPSpace | None = None,
    ):
        r"""Handle multi-input multi-output policy networks.

        Parameters:
            policy (nn.Module): The neural network policy that takes observations and returns actions.
            device (torch.device): The device to run the policy on.
            input_keys (tuple): Keys specifying what fields from the observation to pass to the policy.
            output_keys (tuple): Keys for the fields in the output dictionary that the policy is responsible for.
            spaces (MDPSpace, optional): A utility for managing observation and action spaces, for validation.
        """
        self._policy = policy
        self._end_states = [EpisodeState.TERMINAL, EpisodeState.INTERRUPTED]
        self.device = device
        self.input_keys = input_keys
        self.output_keys = output_keys
        self._spaces = spaces

    def __call__(self, observations: dict[AgentId, DictObservation]) -> dict[AgentId, DictResponse]:
        """Runs the policy and returns the actions."""
        # The network takes observations of size batch x obs for each observation space.
        assert len(observations) > 0, "Observations must not be empty."

        active_agents = [
            agent_id
            for agent_id, obs in observations.items()
            if obs.episode_state not in self._end_states
        ]

        tensor_obs_list = [None] * len(self.input_keys)
        for input_key in self.input_keys:
            np_obs = np.array(
                [observations[agent_id].array_data[input_key] for agent_id in active_agents]
            )

            if self._spaces is not None:
                shape = (np_obs.shape[0],) + self._spaces.state.spaces[input_key].shape
                if shape != np_obs.shape:
                    np_obs = np.reshape(np_obs, shape)

            tensor_obs = torch.tensor(np_obs).to(self.device)
            index = self.input_keys.index(input_key)
            tensor_obs_list[index] = tensor_obs

        outputs: tuple[any, ...] = self._policy(*tensor_obs_list)
        outputs = {key: outputs for i, key in enumerate(self.output_keys)}

        agent_data = [
            (agent_id, DictResponse(list_data={}, scalar_data={})) for agent_id in active_agents
        ]

        for i, (_, response) in enumerate(agent_data):
            for k, data in outputs.items():
                response.list_data[k] = data[i]

        return dict(agent_data)

    @property
    def input_names(self):
        return self.input_keys

    @property
    def output_names(self):
        return self.output_keys

    @property
    def policy(self):
        return self._policy


class QTarget(LoggingMixin, Callback):
    def __init__(
        self,
        *,
        q_net: nn.Module,
        target_q_net: Optional[nn.Module] = None,
        gamma: float = 0.99,
        reward_scale: float = 1.0,
        target_q_tau: float = 0.005,
        data_group: str = "default",
        roll_length: int = 1,
    ):
        """Compute and manage the target Q-values for Q-Learning algorithms.

        Parameters:
            q_net (nn.Module): The Q-network.
            target_q_net (nn.Module, optional): The target Q-network. Defaults to a copy of q_net.
            gamma (float): Discount factor for future rewards.
            reward_scale (float): A scaling factor for the reward values.
            target_q_tau (float): A soft update rate for target Q-network.
            data_group (str): The data group to store the computed Q-target.
            roll_length (int): The rollout length for a batch.

        Methods:
            begin_batch: Compute the target Q-value for a batch.
        """
        super().__init__()
        self._order = 1  # this is to ensure that the data_group is prepared beforehand
        self.data_group = data_group
        self.q_net = q_net
        self.target_q_net = copy.deepcopy(q_net) if target_q_net is None else target_q_net
        self.reward_scale = reward_scale
        self.tau = target_q_tau
        self.rollout_len = roll_length
        self.gamma_matrix = make_gamma_matrix(gamma, self.rollout_len)

    def begin_batch(self, next_observation, rewards, masks):
        next_q_values = self.target_q_net(**next_observation)
        max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
        last_step_masks = split_rollouts(masks, self.rollout_len)[:, -1]
        max_next_q_values = torch.multiply(max_next_q_values, last_step_masks)
        scaled_reward = self.reward_scale * rewards
        scaled_rewards = split_rollouts(scaled_reward, self.rollout_len).squeeze(2)
        q_target = discount(scaled_rewards, max_next_q_values, self.gamma_matrix).detach()

        return {self.data_group: {"q_target": q_target}}

    def end_batch(self):
        super().end_batch()
        soft_update_from_to(self.q_net, self.target_q_net, self.tau)


class QLoss(LossCallback):
    """Compute the Q-Learning loss.

    Parameters:
        name (str): Identifier for this loss component.
        q (nn.Module): The Q-network.
        opt (optim.Optimizer): The optimizer to use for the Q-network.
        lr_schedule (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        max_grad_norm (float): Maximum gradient norm for gradient clipping.
        data_group (str): The data group from which to pull data.
        log_per_param_weights (bool): Whether to log weights per parameter.
        log_per_param_grads (bool): Whether to log gradients per parameter.
    """

    def __init__(
        self,
        *,
        name: str,
        q: nn.Module,
        opt: optim.Optimizer,
        lr_schedule: Optional[optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 10.0,
        data_group: str = "default",
        log_per_param_weights=False,
        log_per_param_grads=False,
    ):
        super().__init__(
            name=name,
            optimizer=opt,
            lr_schedule=lr_schedule,
            network=q,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
            log_per_param_weights=log_per_param_weights,
            log_per_param_grads=log_per_param_grads,
        )
        self.q_network = q
        self.mse = nn.MSELoss()

    def loss(self, observation, q_target, actions):
        indices = actions.to(torch.int64)
        q_value = self.q_network(**observation).gather(1, indices)
        self.log_scalar(f"training/{self.name}_prediction", torch.mean(q_value))
        return self.mse(q_value, q_target)
