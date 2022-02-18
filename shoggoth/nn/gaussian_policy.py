import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.distributions.transforms as transforms
import torch.nn.functional as F
from torch import Tensor


class SquashStretchTransform(transforms.Transform):
    r"""
    Transform via the mapping :math:`y = \alpha \tanh(x/\alpha)`.
    """
    domain = transforms.constraints.real
    bijective = True
    sign = +1

    def __init__(self, tanh_stretch_factor):
        super().__init__()
        self.codomain = transforms.constraints.interval(
            -tanh_stretch_factor, tanh_stretch_factor
        )
        self._stretch = tanh_stretch_factor

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def _call(self, x):
        return self._stretch * torch.tanh(x / self._stretch)

    def _inverse(self, y):
        eps = torch.finfo(y.dtype).eps
        return self._stretch * self.atanh(
            (y / self._stretch).clamp(min=-1.0 + eps, max=1.0 - eps)
        )

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (
            math.log(2.0) - x / self._stretch - F.softplus(-2.0 * x / self._stretch)
        )


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
        tanh_stretch_factor: float = 1.0,
    ):
        super(GaussianPolicyHead, self).__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.normal = dists.MultivariateNormal(
            torch.zeros(self.action_dim), torch.eye(self.action_dim)
        )
        self.squash = SquashStretchTransform(tanh_stretch_factor)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, float]:
        """
        Sample pre-actions and associated negative log-probabilities.

        :return:
            Direct samples (pre-actions) from the policy
            Negative log-probabilities associated to those samples
        """
        bsz, x_dim = x.shape
        assert x_dim == self.hidden_dim
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std)
        raw_sample = self.normal.sample(sample_shape=[bsz])
        log_prob = self.normal.log_prob(raw_sample)
        comp = dists.transforms.ComposeTransform(
            [dists.AffineTransform(mean, std), self.squash]
        )
        sample = comp(raw_sample)
        squash_and_move = dists.TransformedDistribution(self.normal, comp)
        assert sample.shape == (bsz, self.action_dim)
        log_prob = squash_and_move.log_prob(sample).view(bsz, 1)
        assert log_prob.shape == (bsz, 1)
        return sample, -log_prob


class GaussianMLPPolicy(BasePolicy):
    def __init__(self, observation_dim, action_dim, hidden_dims):
        super().__init__()
        self.seq = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip([observation_dim] + hidden_dims, hidden_dims)
            ]
        )
        self.head = GaussianPolicyHead(hidden_dims[-1], action_dim)

    def forward(self, obs):
        x = self.seq(obs)
        pre_actions, neg_log_probs = self.head(x)
        return pre_actions, neg_log_probs
