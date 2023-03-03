from __future__ import annotations

import numpy as np
import torch

from torch import nn, optim

from emote.typing import AgentId, DictObservation, DictResponse, EpisodeState
from emote.utils.gamma_matrix import discount, make_gamma_matrix, split_rollouts

from .callbacks import LoggingCallback, LossCallback


def soft_update_from_to(source, target, tau):  # From rlkit
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class ValueLoss(LossCallback):
    r"""A MSE loss between the action value net and the target q.

    The target q values are not calculated here and need to be added
    to the state before the loss of this module runs.

    :param value (torch.nn.Module): A deep neural net that outputs the discounted loss
        given the current observations.
    :param opt (torch.optim.Optimizer): An optimizer for value.
    :param  lr_schedule (torch.optim.lr_scheduler._LRSchedule): Learning rate schedule
        for the optimizer of value.
    :param max_grad_norm (float): Clip the norm of the gradient during backprop using this value.
    :param name (str): The name of the module. Used e.g. while logging.
    :param data_group (str): The name of the data group from which this Loss takes its data.
    """

    def __init__(
        self,
        *,
        value: nn.Module,
        opt: optim.Optimizer,
        lr_schedule: optim.lr_scheduler._LRScheduler | None = None,
        max_grad_norm: float = 10.0,
        name: str="value",
        data_group: str = "default",
    ):
        super().__init__(
            name=name,
            optimizer=opt,
            lr_schedule=lr_schedule,
            network=value,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
        )
        self.mse = nn.MSELoss()

    def loss(self, observation, rewards_to_go):
        value = self.network(**observation)
        self.log_scalar(f"training/{self.name}_prediction", torch.mean(value))
        return self.mse(value, rewards_to_go)


class NormalizedAdvantage(LoggingCallback):
    r"""Creates rolling averages of the Q nets, and predicts q values using these.

    The module is responsible both for keeping the averages correct in the target q
    networks and supplying q-value predictions using the target q networks.

    :param pi (torch.nn.Module): A deep neural net that outputs actions and their log
        probability given a state.
    :param ln_alpha (torch.tensor): The current weight for the entropy part of the
        soft Q.
    :param q1 (torch.nn.Module): A deep neural net that outputs the discounted loss
        given the current observations and a given action.
    :param q2 (torch.nn.Module): A deep neural net that outputs the discounted loss
        given the current observations and a given action.
    :param gamma (float): Discount factor for the rewards in time.
    :param reward_scale (float): Scale factor for the rewards.
    :param target_q_tau (float): The weight given to the latest network in the
        exponential moving average. So NewTargetQ = OldTargetQ * (1-tau) + Q*tau.
    :param data_group (str): The name of the data group from which this Loss takes its data.
    """

    def __init__(
        self,
        *,
        value: nn.Module,
        gamma: float = 0.99,
        reward_scale: float = 1.0,
        data_group: str = "default",
        device: str = "",
        roll_length: int = 1,
        n_updates_per_iteration = 5,
    ):
        super().__init__(cycle=n_updates_per_iteration)
        self.value = value
        self.data_group = data_group
        self.reward_scale = reward_scale
        self.gamma = torch.tensor(gamma)
        self.rollout_len = roll_length
        self.gamma_matrix = make_gamma_matrix(gamma, self.rollout_len).to(
            device
        )

    def begin_batch(self, rewards, observation):
        assert rewards.shape[1] == self.rollout_len, \
            f"Rewards are shape {rewards.shape[1]}, should be {self.rollout_len}"
        bsz = rewards.shape[0]
        scaled_reward = self.reward_scale * rewards

        scaled_reward = split_rollouts(scaled_reward, self.rollout_len).squeeze(2)
        rtg = discount(
            scaled_reward,
            torch.zeros(scaled_reward.shape[0], 1),
            self.gamma_matrix,
        ).detach()
        assert rtg.shape == (bsz, 1)

        self.log_scalar("training/scaled_reward", torch.mean(scaled_reward))

        value = self.value(**observation).detach()
        advantage = rtg - value
        normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-30)


        return {self.data_group: {"rewards_to_go": rtg, "normalized_advantage": normalized_advantage}}


class PolicyLoss(LossCallback):
    r"""Maximize the soft Q-value for the policy.

    This loss modifies the policy to select the action that gives the highest soft q-value.

    :param pi (torch.nn.Module): A deep neural net that outputs actions and their log
        probability given a state.
    :param ln_alpha (torch.tensor): The current weight for the entropy part of the
        soft Q.
    :param q (torch.nn.Module): A deep neural net that outputs the discounted loss
        given the current observations and a given action.
    :param  lr_schedule (torch.optim.lr_scheduler._LRSchedule): Learning rate schedule
        for the optimizer of policy.
    :param opt (torch.optim.Optimizer): An optimizer for pi.
    :param q2 (torch.nn.Module): A second deep neural net that outputs the discounted
        loss given the current observations and a given action. This is not necessary
        since it is fine if the policy isn't pessimistic, but can be nice for symmetry
        with the Q-loss.
    :param max_grad_norm (float): Clip the norm of the gradient during backprop using this value.
    :param name (str): The name of the module. Used e.g. while logging.
    :param data_group (str): The name of the data group from which this Loss takes its data.
    """

    def __init__(
        self,
        *,
        pi: nn.Module,
        value: nn.Module,
        opt: optim.Optimizer,
        lr_schedule: optim.lr_scheduler._LRScheduler | None = None,
        max_grad_norm: float = 10.0,
        name: str = "policy",
        data_group: str = "default",
    ):
        super().__init__(
            name=name,
            optimizer=opt,
            lr_schedule=lr_schedule,
            network=pi,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
        )
        self.policy = pi
        self.value = value

    def loss(self, observation, normalized_advantage, logp_pi):
        _, current_logp_pi = self.policy(**observation)
        q_pi_min = self.value(**observation)
        # using reparameterization trick
        policy_loss = alpha * logp_pi - q_pi_min
        policy_loss = torch.mean(policy_loss)
        self.log_scalar(f"policy/q_pi_min", torch.mean(q_pi_min))
        self.log_scalar(f"policy/logp_pi", torch.mean(logp_pi))
        assert policy_loss.dim() == 0
        return policy_loss


class FeatureAgentRollout(LoggingCallback):
    """This AgentProxy assumes that the observations will contain flat array of observations names 'obs'"""

    def __init__(self, policy: nn.Module, device: torch.device):
        self.policy = policy
        self._end_states = [EpisodeState.TERMINAL, EpisodeState.INTERRUPTED]
        self.device = device

    def __call__(
        self, observations: dict[AgentId, DictObservation]
    ) -> dict[AgentId, DictResponse]:
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
        actions = self.policy(tensor_obs)[0].detach().cpu().numpy()
        return {
            agent_id: DictResponse(list_data={"actions": actions[i]}, scalar_data={})
            for i, agent_id in enumerate(active_agents)
        }


class VisionAgentProxy:
    """This AgentProxy assumes that the observations will contain image observations 'obs'"""

    def __init__(self, policy: nn.Module, device: torch.device):
        self.policy = policy
        self._end_states = [EpisodeState.TERMINAL, EpisodeState.INTERRUPTED]
        self.device = device

    def __call__(
        self, observations: dict[AgentId, DictObservation]
    ) -> dict[AgentId, DictResponse]:
        """Runs the policy and returns the actions."""
        # The network takes observations of size batch x obs for each observation space.
        assert len(observations) > 0, "Observations must not be empty."
        active_agents = [
            agent_id
            for agent_id, obs in observations.items()
            if obs.episode_state not in self._end_states
        ]
        np_obs = np.array(
            [observations[agent_id].array_data["obs"] for agent_id in active_agents]
        )
        tensor_obs = torch.tensor(np_obs).to(self.device)
        actions = self.policy(tensor_obs)[0].detach().cpu().numpy()
        return {
            agent_id: DictResponse(list_data={"actions": actions[i]}, scalar_data={})
            for i, agent_id in enumerate(active_agents)
        }
