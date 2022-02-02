import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .callbacks import LoggingCallback, LossCallback


def soft_update_from_to(source, target, tau):  # From rlkit
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class QLoss(LossCallback):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        name: str,
        q_forward: nn.Module,
        max_grad_norm: float,
    ):
        super().__init__(optimizer, max_grad_norm, name)
        self.q_forward = q_forward
        self.mse = nn.MSELoss()

    def loss(self, obs, actions, q_target, **kwargs):
        q_value = self.q_forward(obs, actions)
        self.scalar_log(f"training/{self.name}_prediction", torch.mean(q_value))
        return self.mse(q_value, q_target)


class QTarget(LoggingCallback):
    def __init__(self, net, gamma_np, reward_scale, target_q_tau):
        super().__init__()
        self.net = net
        self.reward_scale = reward_scale
        self.tau = target_q_tau
        self.gamma = torch.tensor(gamma_np)

    def begin_training(self, **kwargs):
        self.q1_vars = self.net.q1.parameters()
        self.q2_vars = self.net.q2.parameters()
        self.q1_target_vars = self.net.q1_target.parameters()
        self.q2_target_vars = self.net.q2_target.parameters()

    def begin_backward(self, next_obs, rewards, masks, **kwargs):
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

        self.scalar_log("training/next_logp_pi", torch.mean(next_logp_pi))
        self.scalar_log("training/min_next_q_target", torch.mean(min_next_q_target))
        self.scalar_log("training/scaled_reward", torch.mean(scaled_reward))
        self.scalar_log("training/q_target", torch.mean(q_target))

        return {"q_target": q_target}

    def end_backward(self, **kwargs):
        soft_update_from_to(self.net.q1, self.net.q1_target, self.tau)
        soft_update_from_to(self.net.q2, self.net.q2_target, self.tau)


class PolicyLoss(LossCallback):
    def __init__(
        self,
        optimizer,
        name: str,
        net,
        max_grad_norm: float,
    ):
        super().__init__(optimizer, max_grad_norm, name)
        self.net = net
        self._log_alpha_vars = self.net.log_alpha_vars

    def loss(self, obs, **kwargs):
        assert isinstance(obs, tuple) and len(obs) > 0
        p_sample, logp_pi = self.net.policy(obs)
        q_pi_min = torch.min(self.net.q1(obs, p_sample), self.net.q2(obs, p_sample))
        self.scalar_log(f"policy/q_pi_min", torch.mean(q_pi_min))
        self.scalar_log(f"policy/logp_pi", torch.mean(logp_pi))
        # using reparameterization trick
        alpha = torch.exp(self._log_alpha_vars[0].detach())
        self.scalar_log(f"policy/alpha", torch.mean(alpha))
        policy_loss = alpha * logp_pi - q_pi_min
        policy_loss = torch.mean(policy_loss)
        assert policy_loss.dim() == 0
        return policy_loss


class AlphaLoss(LossCallback):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        name: str,
        net,
        n_actions: int,
        entropy_eps: float,
        max_alpha: float,
        max_grad_norm: float,
    ):
        super().__init__(optimizer, max_grad_norm, name)
        self.net = net
        self._max_log_alpha = np.log(max_alpha)
        # TODO(singhblom) Check number of actions
        # self._target_entropy = -np.prod(self.env.action_space.shape).item()  # Value from rlkit from Harnouja
        self._target_entropy = (
            n_actions * (1.0 + np.log(2.0 * np.pi * entropy_eps**2)) / 2.0
        )
        self.vars = self.net.log_alpha_vars  # This is log(alpha)

    def loss(self, obs, **kwargs):
        assert isinstance(obs, tuple) and len(obs) > 0
        bsz, *_others = obs[0].shape
        _p_sample, logp_pi = self.net.policy(obs)
        alpha_loss = -torch.mean(
            self.vars[0] * (logp_pi + self._target_entropy).detach()
        )
        assert alpha_loss.dim() == 0
        self.scalar_log("loss/alpha_loss", alpha_loss)
        return alpha_loss

    def end_backward(self, **kwargs):
        self.vars[0].requires_grad_(False)
        self.vars[0] = torch.clamp_max_(self.vars[0], self._max_log_alpha)
        self.vars[0].requires_grad_(True)
        self.scalar_log("training/alpha_value", torch.exp(self.vars[0]).item())
        super().end_backward(**kwargs)
