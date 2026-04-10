from functools import partial

import torch
import torch.distributions as dists
import torch.distributions.transforms as transforms
import torch.nn as nn

from torch import Tensor

from emote.nn.action_value_mlp import GaussianPolicyHead
from emote.nn.initialization import xavier_uniform_init_
from emote.nn.trunks import Trunk


class HybridGaussianBernoulliPolicy(nn.Module):
    def __init__(
        self,
        perception_model: nn.Module,
        trunk: Trunk,
        num_continuous_heads: int,
        num_boolean_heads: int,
        continuous_entropy_weight: float,
        bernoulli_entropy_weight: float,
        chunking_size: int,
    ) -> None:
        super().__init__()

        # perception model
        self.perception_model = perception_model

        self.trunk = trunk

        self._num_continuous_heads = num_continuous_heads
        self._num_boolean_heads = num_boolean_heads
        self._chunking_size = chunking_size

        # Gaussian policy head
        self.gaussian_head = GaussianPolicyHead(
            hidden_dim=trunk.output_dim(),
            action_dim=num_continuous_heads * chunking_size,
        )
        self.gaussian_head.apply(partial(xavier_uniform_init_, gain=1.0))

        # Boolean policy head
        if num_boolean_heads > 0:
            self.boolean_head = nn.Linear(trunk.output_dim(), num_boolean_heads * chunking_size)
            self.boolean_head.apply(partial(xavier_uniform_init_, gain=1.0))
        else:
            self.boolean_head = None

        self._entropy_const = (
            (num_continuous_heads * chunking_size)
            * (1.0 + torch.log(torch.tensor(2.0 * torch.pi)))
            / 2.0
        )
        self.continuous_entropy_weight = continuous_entropy_weight
        self.bernoulli_entropy_weight = bernoulli_entropy_weight

    def forward(
        self,
        features: Tensor,
        **kwargs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        perception_output = self.perception_model.forward(**kwargs)
        trunk_output = self.trunk(torch.cat([perception_output, features], dim=1))
        gaussian_mean, gaussian_std = self.gaussian_head(trunk_output)

        if self.boolean_head is not None:
            boolean_logit = self.boolean_head(trunk_output)
            return gaussian_mean, gaussian_std, boolean_logit

        return gaussian_mean, gaussian_std

    def sample_actions(self, *args, **kwargs) -> Tensor:
        return self._sample_actions_and_compute_entropy(compute_entropy=False, *args, **kwargs)

    def sample_actions_and_entropies(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        return self._sample_actions_and_compute_entropy(compute_entropy=True, *args, **kwargs)

    def entropy(self, *args, **kwargs) -> Tensor:
        model_output = self.forward(*args, **kwargs)
        gaussian_mean, gaussian_std, boolean_logit = (
            model_output if len(model_output) == 3 else (*model_output, None)
        )
        return self._compute_entropy(gaussian_mean, gaussian_std, boolean_logit)

    def _sample_actions_and_compute_entropy(
        self, compute_entropy: bool = False, *args, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        model_output = self.forward(*args, **kwargs)
        gaussian_mean, gaussian_std, boolean_logit = (
            model_output if len(model_output) == 3 else (*model_output, None)
        )

        actions = self._sample_actions(gaussian_mean, gaussian_std, boolean_logit)

        if compute_entropy:
            entropy = self._compute_entropy(gaussian_mean, gaussian_std, boolean_logit)
            return actions, entropy

        return actions

    def _sample_actions(
        self, gaussian_mean: Tensor, gaussian_std: Tensor, boolean_logit: Tensor | None
    ) -> Tensor:
        # continuous distribution and sampling
        dist = dists.TransformedDistribution(
            dists.Independent(dists.Normal(gaussian_mean, gaussian_std), 1),
            transforms.TanhTransform(cache_size=1),
        )
        continuous_sample = dist.rsample()

        # boolean distribution and sampling
        if boolean_logit is not None:
            u = torch.rand_like(boolean_logit)
            gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            boolean_sample = ((boolean_logit + gumbel_noise) > 0).float()

            return torch.cat([continuous_sample, boolean_sample], dim=-1)

        return continuous_sample

    def _compute_entropy(
        self, gaussian_mean: Tensor, gaussian_std: Tensor, boolean_logit: Tensor | None
    ) -> Tensor:
        mean_tanh = torch.tanh(gaussian_mean)
        correction_term = torch.log(1 - mean_tanh**2).sum(dim=-1).view(-1, 1)
        continuous_entropy = (
            torch.log(gaussian_std).sum(dim=-1).view(-1, 1) + correction_term + self._entropy_const
        )

        if boolean_logit is not None:
            p = torch.sigmoid(boolean_logit)
            eps = 1e-8
            bernoulli_entropy = (
                (-p * torch.log(p + eps) - (1.0 - p) * torch.log(1.0 - p + eps))
                .sum(dim=-1)
                .view(-1, 1)
            )

            return (
                self.continuous_entropy_weight * continuous_entropy
                + self.bernoulli_entropy_weight * bernoulli_entropy
            )

        return self.continuous_entropy_weight * continuous_entropy

    def get_action_dim(self) -> tuple[int, int]:
        return self._num_continuous_heads, self._num_boolean_heads

    def get_chunking_size(self) -> int:
        return self._chunking_size

    def get_trainable_params(self):
        if self.boolean_head is not None:
            return (
                list(self.trunk.parameters())
                + list(self.gaussian_head.parameters())
                + list(self.boolean_head.parameters())
            )
        else:
            return list(self.trunk.parameters()) + list(self.gaussian_head.parameters())
