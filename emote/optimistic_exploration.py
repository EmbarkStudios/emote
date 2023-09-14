# Adapted from https://github.com/microsoft/oac-explore/blob/master/optimistic_exploration.py (MIT license)

import math

import functorch
import torch
import torch.distributions as dists
import torch.distributions.transforms as transforms


def get_optimistic_exploration_action_batch(obs, mean, std, q1, q2, beta_ub, delta):
    assert obs[0].ndim == 2
    assert mean.ndim == 2
    assert std.ndim == 2

    mean.requires_grad_()

    def q_ub_fn(mean_sample):
        actions_sample = torch.tanh(mean_sample)
        q1_sample = q1(actions_sample, *obs)
        q2_sample = q2(actions_sample, *obs)
        q_mean_sample = (q1_sample + q2_sample) / 2.0
        sigma_q_sample = torch.abs(q1_sample - q2_sample) / 2.0
        q_ub_sample = q_mean_sample + beta_ub * sigma_q_sample

        return q_ub_sample

    # dims = B x 1 x B x N
    per_sample_grads = torch.autograd.functional.jacobian(q_ub_fn, mean, vectorize=True)

    # There's only one non-zero row per sample
    # By summing along 2, for each sample we get 1 x N (dims = B x 1 x N)
    per_sample_grads = torch.sum(per_sample_grads, dim=2)

    # Squeeze to get dims = B x N
    per_sample_grads = per_sample_grads.squeeze(1)
    assert per_sample_grads is not None
    assert per_sample_grads.shape == mean.shape

    Sigma_T = torch.pow(std, 2)

    denom = (
        torch.sqrt(
            torch.sum(
                torch.mul(torch.pow(per_sample_grads, 2), Sigma_T),
                dim=tuple(range(1, per_sample_grads.dim())),
            )
        )
        + 10e-6
    )

    mean_change = (
        math.sqrt(2.0 * delta)
        * torch.mul(Sigma_T, per_sample_grads)
        / denom.unsqueeze(1)
    )
    assert mean_change.shape == mean.shape

    mean_exploration = mean + mean_change
    assert mean_exploration.shape == std.shape

    # Construct the tanh normal distribution and sample the exploratory action from it
    dist = dists.TransformedDistribution(
        dists.Independent(dists.Normal(mean_exploration, std), 1),
        transforms.TanhTransform(cache_size=1),
    )

    ac = dist.sample()

    return ac
