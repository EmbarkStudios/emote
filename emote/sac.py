import copy
import random
import time

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from emote.proxies import AgentProxy
from emote.typing import AgentId, DictObservation, DictResponse, EpisodeState
from emote.utils.gamma_matrix import discount, make_gamma_matrix, split_rollouts

from .callback import Callback
from .callbacks import LoggingMixin, LossCallback


def soft_update_from_to(source, target, tau):  # From rlkit
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class QLoss(LossCallback):
    r"""A MSE loss between the action value net and the target q.

    The target q values are not calculated here and need to be added
    to the state before the loss of this module runs.

    :param name (str): The name of the module. Used e.g. while logging.
    :param q (torch.nn.Module): A deep neural net that outputs the discounted loss
        given the current observations and a given action.
    :param opt (torch.optim.Optimizer): An optimizer for q.
    :param  lr_schedule (torch.optim.lr_scheduler._LRSchedule): Learning rate schedule
        for the optimizer of q.
    :param max_grad_norm (float): Clip the norm of the gradient during backprop using this value.
    :param data_group (str): The name of the data group from which this Loss takes its data.
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
    ):
        super().__init__(
            name=name,
            optimizer=opt,
            lr_schedule=lr_schedule,
            network=q,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
        )
        self.q_network = q
        self.mse = nn.MSELoss()

    def loss(self, observation, actions, q_target):
        q_value = self.q_network(actions, **observation)
        self.log_scalar(f"training/{self.name}_prediction", torch.mean(q_value))
        return self.mse(q_value, q_target)


class QTarget(LoggingMixin, Callback):
    r"""Creates rolling averages of the Q nets, and predicts q values using these.

    The module is responsible both for keeping the averages correct in the target q
    networks and supplying q-value predictions using the target q networks.

    :param pi (torch.nn.Module): A deep neural net that outputs actions and their log
        probability given a state.
    :param ln_alpha (torch.tensor): The current weight for the entropy part of the
        soft Q.
    :param q (List[torch.nn.Module]): A list of deep neural nets that output the discounted loss
        given the current observations and a given action.
    :param qt (List[torch.nn.Module]): A list of target networks.
    :param gamma (float): Discount factor for the rewards in time.
    :param reward_scale (float): Scale factor for the rewards.
    :param target_q_tau (float): The weight given to the latest network in the
        exponential moving average. So NewTargetQ = OldTargetQ * (1-tau) + Q*tau.
    :param data_group (str): The name of the data group from which this Loss takes its data.
    :param roll_length (int): Rollout length. (default 1)
    """

    def __init__(
        self,
        *,
        pi: nn.Module,
        ln_alpha: torch.tensor,
        q: List[nn.Module],
        qt: Optional[List[nn.Module]] = None,
        gamma: float = 0.99,
        reward_scale: float = 1.0,
        target_q_tau: float = 0.005,
        data_group: str = "default",
        roll_length: int = 1,
    ):
        super().__init__()
        self._order = 1  # this is to ensure that the data_group is prepared beforehand
        self.data_group = data_group
        self.policy = pi
        self.qt = copy.deepcopy(q) if qt is None else qt
        self.ln_alpha = ln_alpha
        self.q = q
        self.reward_scale = reward_scale
        self.tau = target_q_tau
        self.gamma = torch.tensor(gamma)
        self.rollout_len = roll_length
        self.gamma_matrix = make_gamma_matrix(gamma, self.rollout_len).to(
            ln_alpha.device
        )

    def begin_batch(self, next_observation, rewards, masks):
        next_p_sample, next_logp_pi = self.policy(**next_observation)

        sampled_indices = np.random.choice(len(self.q), 2, replace=False)
        next_q1t = self.qt[sampled_indices[0]](next_p_sample, **next_observation)
        next_q2t = self.qt[sampled_indices[1]](next_p_sample, **next_observation)
        min_next_qt = torch.min(next_q1t, next_q2t)
        bsz = rewards.shape[0]

        alpha = torch.exp(self.ln_alpha)
        next_value = min_next_qt - alpha * next_logp_pi
        scaled_reward = self.reward_scale * rewards

        last_step_masks = split_rollouts(masks, self.rollout_len)[:, -1]
        scaled_reward = split_rollouts(scaled_reward, self.rollout_len).squeeze(2)
        next_value_masked = torch.multiply(next_value, last_step_masks)
        qt = discount(scaled_reward, next_value_masked, self.gamma_matrix).detach()
        assert qt.shape == (bsz, 1)

        self.log_scalar("training/next_logp_pi", torch.mean(next_logp_pi))
        self.log_scalar("training/min_next_q_target", torch.mean(min_next_qt))
        self.log_scalar("training/scaled_reward", torch.mean(scaled_reward))
        self.log_scalar("training/q_target", torch.mean(qt))

        return {self.data_group: {"q_target": qt}}

    def end_batch(self):
        super().end_batch()
        for i in range(len(self.q)):
            soft_update_from_to(self.q[i], self.qt[i], self.tau)


class PolicyLoss(LossCallback):
    r"""Maximize the soft Q-value for the policy.

    This loss modifies the policy to select the action that gives the highest soft q-value.

    :param pi (torch.nn.Module): A deep neural net that outputs actions and their log
        probability given a state.
    :param ln_alpha (torch.tensor): The current weight for the entropy part of the
        soft Q.
    :param q (Union[List[torch.nn.Module], torch.nn.Module]): One or more deep neural nets that output the discounted loss
        given the current observations and a given action.
    :param  lr_schedule (torch.optim.lr_scheduler._LRSchedule): Learning rate schedule
        for the optimizer of policy.
    :param opt (torch.optim.Optimizer): An optimizer for pi.
    :param max_grad_norm (float): Clip the norm of the gradient during backprop using this value.
    :param name (str): The name of the module. Used e.g. while logging.
    :param data_group (str): The name of the data group from which this Loss takes its data.
    :param pessimistic (bool): A flag indicating whether the policy updates should be pessimistic (take a minimum of multiple Q networks).
    """

    def __init__(
        self,
        *,
        pi: nn.Module,
        ln_alpha: torch.tensor,
        q: Union[List[nn.Module], nn.Module],
        opt: optim.Optimizer,
        lr_schedule: Optional[optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 10.0,
        name: str = "policy",
        data_group: str = "default",
        pessimistic: bool = False,
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
        self._ln_alpha = ln_alpha
        self.q = q
        self.pessimistic = pessimistic

    def loss(self, observation):
        p_sample, logp_pi = self.policy(**observation)

        if isinstance(self.q, nn.Module):
            q_pi = self.q(p_sample, **observation)
        elif isinstance(self.q, List):
            assert len(self.q) >= 2
            q_pis = torch.cat([q(p_sample, **observation) for q in self.q], 1)

            if self.pessimistic:
                q_pi, _ = torch.min(q_pis, dim=1, keepdim=True)
            else:
                q_pi = torch.mean(q_pis, dim=1, keepdim=True)

        # using reparameterization trick
        alpha = torch.exp(self._ln_alpha).detach()
        policy_loss = alpha * logp_pi - q_pi
        policy_loss = torch.mean(policy_loss)
        self.log_scalar("policy/q_pi", torch.mean(q_pi))
        self.log_scalar("policy/logp_pi", torch.mean(logp_pi))
        self.log_scalar("policy/alpha", torch.mean(alpha))
        assert policy_loss.dim() == 0
        return policy_loss


class AlphaLoss(LossCallback):
    r"""Tweaks the alpha so that a specific target entropy is kept.

    The target entropy is scaled with the number of actions and a provided entropy
    scaling factor.

    :param pi (torch.nn.Module): A deep neural net that outputs actions and their log
        probability given a state.
    :param ln_alpha (torch.tensor): The current weight for the entropy part of the
        soft Q.
    :param  lr_schedule (torch.optim.lr_scheduler._LRSchedule): Learning rate schedule
        for the optimizer of alpha.
    :param opt (torch.optim.Optimizer): An optimizer for ln_alpha.
    :param n_actions (int): The dimension of the action space. Scales the target
        entropy.
    :param max_grad_norm (float): Clip the norm of the gradient during backprop using
        this value.
    :param entropy_eps (float): Scaling value for the target entropy.
    :param name (str): The name of the module. Used e.g. while logging.
    :param data_group (str): The name of the data group from which this Loss takes its
        data.
    """

    def __init__(
        self,
        *,
        pi: nn.Module,
        ln_alpha: torch.tensor,
        opt: optim.Optimizer,
        lr_schedule: Optional[optim.lr_scheduler._LRScheduler] = None,
        n_actions: int,
        max_grad_norm: float = 10.0,
        entropy_eps: float = 0.089,
        max_alpha: float = 0.2,
        name: str = "alpha",
        data_group: str = "default",
    ):
        super().__init__(
            name=name,
            optimizer=opt,
            lr_schedule=lr_schedule,
            network=None,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
        )
        self.policy = pi
        self._max_ln_alpha = torch.log(torch.tensor(max_alpha, device=ln_alpha.device))
        # TODO(singhblom) Check number of actions
        # self.t_entropy = -np.prod(self.env.action_space.shape).item()  # Value from rlkit from Harnouja
        self.t_entropy = (
            n_actions * (1.0 + np.log(2.0 * np.pi * entropy_eps**2)) / 2.0
        )
        self.ln_alpha = ln_alpha  # This is log(alpha)

    def loss(self, observation):
        with torch.no_grad():
            _, logp_pi = self.policy(**observation)
            entropy = -logp_pi
            error = entropy - self.t_entropy
        alpha_loss = torch.mean(self.ln_alpha * error.detach())
        assert alpha_loss.dim() == 0
        self.log_scalar("loss/alpha_loss", alpha_loss)
        self.log_scalar("training/entropy", torch.mean(entropy).item())
        return alpha_loss

    def end_batch(self):
        super().end_batch()
        self.ln_alpha.requires_grad_(False)
        self.ln_alpha = torch.clamp_max_(self.ln_alpha, self._max_ln_alpha)
        self.ln_alpha.requires_grad_(True)
        self.log_scalar("training/alpha_value", torch.exp(self.ln_alpha).item())
        self.log_scalar("training/target_entropy", self.t_entropy)

    def state_dict(self):
        state = super().state_dict()
        state["network_state_dict"] = self.ln_alpha

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.ln_alpha = state_dict.pop("network_state_dict")
        # TODO(singhblom) Set the right device
        super().load_state_dict(state_dict)


class AgentProxyWrapper:
    def __init__(self, *, inner: AgentProxy, **kwargs):
        super().__init__(**kwargs)
        self._inner = inner

    def __call__(self, *args, **kwargs):
        self._inner(*args, **kwargs)

    @property
    def input_names(self):
        return self._inner.input_names


class FeatureAgentProxy:
    """An agent proxy for basic MLPs.

    This AgentProxy assumes that the observations will contain a single flat array of features.
    """

    def __init__(self, policy: nn.Module, device: torch.device, input_key: str = "obs"):
        """Create a new proxy.

        :param policy: The policy to execute for actions.
        :param device: The device to run on.
        :param input_key: The name of the features. (default: "obs")
        """
        self.policy = policy
        self._end_states = [EpisodeState.TERMINAL, EpisodeState.INTERRUPTED]
        self.device = device

        self._input_key = input_key

    def __call__(
        self,
        observations: Dict[AgentId, DictObservation],
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
                [
                    observations[agent_id].array_data[self._input_key]
                    for agent_id in active_agents
                ]
            )
        ).to(self.device)

        actions = self.policy(tensor_obs)[0].detach().cpu().numpy()

        return {
            agent_id: DictResponse(list_data={"actions": actions[i]}, scalar_data={})
            for i, agent_id in enumerate(active_agents)
        }

    @property
    def input_names(self):
        return (self._input_key,)


class VisionAgentProxy:
    """This AgentProxy assumes that the observations will contain image observations 'obs'"""

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
        np_obs = np.array(
            [observations[agent_id].array_data["obs"] for agent_id in active_agents]
        )
        tensor_obs = torch.tensor(np_obs).to(self.device)
        actions = self.policy(tensor_obs)[0].detach().cpu().numpy()
        return {
            agent_id: DictResponse(list_data={"actions": actions[i]}, scalar_data={})
            for i, agent_id in enumerate(active_agents)
        }
