from functools import partial
from typing import Tuple

import torch
import torch.distributions as dists
import torch.distributions.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from emote.nn.initialization import ortho_init_, xavier_uniform_init_


class RobustTanhTransform(transforms.TanhTransform):
    def _inverse(self, y):
        eps = torch.finfo(y.dtype).eps
        input_val = y.clamp(min=-1.0 + eps, max=1.0 - eps)
        return torch.atanh(input_val)


class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def post_process(self, actions):
        """Post-process a pre-action into a post-action"""
        return actions

    def infer(self, x: Tensor):
        """
        Samples pre-actions and associated post-actions (actual decisions) from the policy given the
        encoder input. Only for use at inference time; defaults to identity transformation. Crucial to reimplement for
        discrete reparametrized policies.
        """
        p_samp, _ = self(x)
        return p_samp, self.post_process(p_samp)


class GaussianPolicyHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
    ):
        super(GaussianPolicyHead, self).__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, float]:
        """
        Sample pre-actions and associated log-probabilities.

        :return:
            Direct samples (pre-actions) from the policy
            log-probabilities associated to those samples
        """
        bsz, _ = x.shape

        mean = self.mean(x).clamp(min=-5, max=5)  # equates to 0.99991 after tanh.
        std = torch.exp(self.log_std(x).clamp(min=-20, max=2))

        dist = dists.TransformedDistribution(
            dists.Independent(dists.Normal(mean, std), 1),
            RobustTanhTransform(),
        )
        sample = dist.rsample()
        log_prob = dist.log_prob(sample).view(bsz, 1)

        assert sample.shape == (bsz, self.action_dim)
        assert log_prob.shape == (bsz, 1)

        return sample, log_prob


class GaussianMlpPolicy(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip([observation_dim] + hidden_dims, hidden_dims)
            ],
        )
        self.policy = GaussianPolicyHead(hidden_dims[-1], action_dim)

        self.encoder.apply(ortho_init_)
        self.policy.apply(partial(xavier_uniform_init_, gain=0.01))

    def forward(self, obs):
        return self.policy(self.encoder(obs))
