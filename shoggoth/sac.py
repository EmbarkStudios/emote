from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch import optim

from shoggoth.proxies import Observations, Responses

from .callbacks import LoggingCallback, LossCallback


@dataclass
class SACNetwork:
    q1: nn.Module
    q2: nn.Module
    q1_target: nn.Module
    q2_target: nn.Module
    policy: nn.Module
    log_alpha_vars: torch.TensorType


def soft_update_from_to(source, target, tau):  # From rlkit
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class QLoss(LossCallback):
    def __init__(
        self,
        name: str,
        optimizer: optim.Optimizer,
        q_forward: nn.Module,
        max_grad_norm: float = 10.0,
    ):
        super().__init__(name, optimizer, max_grad_norm)
        self.q_forward = q_forward
        self.mse = nn.MSELoss()

    def loss(self, obs, actions, q_target):
        q_value = self.q_forward(obs, actions)
        self.log_scalar(f"training/{self.name}_prediction", torch.mean(q_value))
        return self.mse(q_value, q_target)


class QTarget(LoggingCallback):
    def __init__(
        self,
        net: SACNetwork,
        gamma: float = 0.99,
        reward_scale: float = 1.0,
        target_q_tau: float = 0.005,
    ):
        super().__init__()
        self.net = net
        self.reward_scale = reward_scale
        self.tau = target_q_tau
        self.gamma = torch.tensor(gamma)

    def begin_batch(self, next_obs, rewards, masks):
        bsz, *_ = next_obs[0].shape
        next_p_sample, next_logp_pi = self.net.policy(next_obs)
        next_q1_target = self.net.q1_target(next_obs, next_p_sample)
        next_q2_target = self.net.q2_target(next_obs, next_p_sample)
        min_next_q_target = torch.min(next_q1_target, next_q2_target)

        alpha = torch.exp(self.net.log_alpha_vars[0])
        next_value = min_next_q_target - alpha * next_logp_pi

        scaled_reward = self.reward_scale * rewards
        q_target = (scaled_reward + self.gamma * masks * next_value).detach()
        assert q_target.shape == (bsz, 1)

        self.log_scalar("training/next_logp_pi", torch.mean(next_logp_pi))
        self.log_scalar("training/min_next_q_target", torch.mean(min_next_q_target))
        self.log_scalar("training/scaled_reward", torch.mean(scaled_reward))
        self.log_scalar("training/q_target", torch.mean(q_target))

        return {"q_target": q_target}

    def end_batch(self):
        super().end_batch()
        soft_update_from_to(self.net.q1, self.net.q1_target, self.tau)
        soft_update_from_to(self.net.q2, self.net.q2_target, self.tau)


class PolicyLoss(LossCallback):
    def __init__(
        self,
        name: str,
        optimizer: optim.Optimizer,
        net: SACNetwork,
        max_grad_norm: float = 10.0,
    ):
        super().__init__(name, optimizer, max_grad_norm)
        self.net = net
        self._log_alpha_vars = self.net.log_alpha_vars

    def loss(self, obs):
        assert isinstance(obs, tuple) and len(obs) > 0
        p_sample, logp_pi = self.net.policy(obs)
        q_pi_min = torch.min(self.net.q1(obs, p_sample), self.net.q2(obs, p_sample))
        # using reparameterization trick
        alpha = torch.exp(self._log_alpha_vars[0].detach())
        policy_loss = alpha * logp_pi - q_pi_min
        policy_loss = torch.mean(policy_loss)
        self.log_scalar(f"policy/q_pi_min", torch.mean(q_pi_min))
        self.log_scalar(f"policy/logp_pi", torch.mean(logp_pi))
        self.log_scalar(f"policy/alpha", torch.mean(alpha))
        assert policy_loss.dim() == 0
        return policy_loss


class AlphaLoss(LossCallback):
    def __init__(
        self,
        name: str,
        optimizer: optim.Optimizer,
        net: SACNetwork,
        n_actions: int,
        max_grad_norm: float = 10.0,
        entropy_eps: float = 0.089,
        max_alpha: float = 0.2,
    ):
        super().__init__(name, optimizer, max_grad_norm)
        self.net = net
        self._max_log_alpha = np.log(max_alpha)
        # TODO(singhblom) Check number of actions
        # self._target_entropy = -np.prod(self.env.action_space.shape).item()  # Value from rlkit from Harnouja
        self._target_entropy = (
            n_actions * (1.0 + np.log(2.0 * np.pi * entropy_eps**2)) / 2.0
        )
        self.vars = self.net.log_alpha_vars  # This is log(alpha)

    def loss(self, obs):
        assert isinstance(obs, tuple) and len(obs) > 0
        bsz, *_others = obs[0].shape
        _p_sample, logp_pi = self.net.policy(obs)
        alpha_loss = -torch.mean(
            self.vars[0] * (logp_pi + self._target_entropy).detach()
        )
        assert alpha_loss.dim() == 0
        self.log_scalar("loss/alpha_loss", alpha_loss)
        return alpha_loss

    def end_batch(self):
        super().end_batch()
        self.vars[0].requires_grad_(False)
        self.vars[0] = torch.clamp_max_(self.vars[0], self._max_log_alpha)
        self.vars[0].requires_grad_(True)
        self.log_scalar("training/alpha_value", torch.exp(self.vars[0]).item())


class FeatureAgentProxy(object):
    def __init__(self, network: SACNetwork):
        super().__init__()
        self.network = network

    def __call__(self, observations: Observations) -> Responses:
        # The network takes observations of size batch x obs for each observation space.
        assert "obs" in observations, "Observations must have key 'obs'"
        assert len(observations["obs"]) > 0, "Observations are empty."
        a2o = observations["obs"]  # Agent to Observation
        obs = (torch.stack([torch.tensor(data) for _, data in a2o]),)
        actions = self.network.policy(obs)

        return {
            "action": {agent_id: actions[i]} for i, agent_id in enumerate(a2o.keys())
        }
