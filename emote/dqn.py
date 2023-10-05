

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
from emote.sac import soft_update_from_to
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
        """Create a new proxy.

        :param policy (nn.Module): The policy to invoke
        :param device (torch.device): The device to run on
        :param input_keys (tuple): The names of the inputs to the policy
        :param output_keys (tuple): The names of the outputs of the policy
        :param spaces (MDPSpace): The spaces of the inputs and outputs
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
        # target_q_tau: float = 0.005,
        data_group: str = "default",
        roll_length: int = 1,
    ):
        super().__init__()
        self._order = 1  # this is to ensure that the data_group is prepared beforehand
        self.data_group = data_group
        self.q_net = q_net
        self.target_q_net = copy.deepcopy(q_net) if target_q_net is None else target_q_net
        self.reward_scale = reward_scale
        # self.tau = target_q_tau
        self.rollout_len = roll_length
        self.gamma_matrix = make_gamma_matrix(gamma, self.rollout_len)
        self.step_counter = 0

    def begin_batch(self, next_observation, rewards):
        next_q_values = self.target_q_net(**next_observation)
        max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
        scaled_reward = self.reward_scale * rewards
        scaled_rewards = split_rollouts(scaled_reward, self.rollout_len).squeeze(2)
        q_target = discount(scaled_rewards, max_next_q_values, self.gamma_matrix).detach()
        
        return {self.data_group: {"q_target": q_target}}
    
    def end_batch(self):
        super().end_batch()
        self.step_counter += 1
        if self.step_counter % 500 == 0 and self.step_counter > 0:
            self.target_q_net = copy.deepcopy(self.q_net)
            print("Updated target network at step ", self.step_counter)

        # soft_update_from_to(self.q_net, self.target_q_net, self.tau)


class QLoss(LossCallback):
    r"""
    :param name (str): The name of the module. Used e.g. while logging.
    :param q (torch.nn.Module): A deep neural net that outputs the discounted loss
        given the current observations and a given action.
    :param opt (torch.optim.Optimizer): An optimizer for q.
    :param  lr_schedule (torch.optim.lr_scheduler._LRSchedule): Learning rate schedule
        for the optimizer of q.
    :param max_grad_norm (float): Clip the norm of the gradient during backprop using this value.
    :param data_group (str): The name of the data group from which this Loss takes its data.
    :param log_per_param_weights (bool): If true, log each individual policy parameter that is optimized (norm and value histogram).
    :param log_per_param_grads (bool): If true, log the gradients of each individual policy parameter that is optimized (norm and histogram).
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

    # TODO: Luc: Move this and sac to emote/algorithms/

    def loss(self, observation, q_target, actions):
        indices = actions.to(torch.int64)
        indices = indices.argmax(dim=1).unsqueeze(1)
        q_value = self.q_network(**observation).gather(1, indices)
        self.log_scalar(f"training/{self.name}_prediction", torch.mean(q_value))
        # out = self.mse(q_value, q_target)
        # print("QLOSS", out)
        # return out
        return self.mse(q_value, q_target)