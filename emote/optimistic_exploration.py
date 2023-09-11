# Adapted from https://github.com/microsoft/oac-explore/blob/master/optimistic_exploration.py (MIT license)

import math

import torch
import torch.distributions as dists
import torch.distributions.transforms as transforms


def get_optimistic_exploration_action_batch(obs, mean, std, q1, q2, beta_ub, delta):
    # Ensure that obs, mean, std are batched
    assert obs[0].ndim == 2
    assert len(list(mean.shape)) == 2
    assert len(list(std.shape)) == 2

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
    per_sample_grads = torch.autograd.functional.jacobian(q_ub_fn, mean)

    # there's only one non-zero row per sample
    # by summing along 2, for each sample we get 1 x N (dims = B x 1 x N)
    per_sample_grads = torch.sum(per_sample_grads, dim=2)

    # squeeze to get dims = B x N
    per_sample_grads = per_sample_grads.squeeze(1)

    assert per_sample_grads is not None
    assert mean.shape == per_sample_grads.shape

    # Obtain Sigma_T (the covariance of the normal distribution)
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

    # Obtain the change in mu
    mean_change = (
        math.sqrt(2.0 * delta)
        * torch.mul(Sigma_T, per_sample_grads)
        / denom.unsqueeze(1)
    )

    assert mean_change.shape == mean.shape

    mean_exploration = mean + mean_change

    # Construct the tanh normal distribution and sample the exploratory action from it
    assert mean_exploration.shape == std.shape

    dist = dists.TransformedDistribution(
        dists.Independent(dists.Normal(mean_exploration, std), 1),
        transforms.TanhTransform(cache_size=1),
    )

    ac = dist.sample()

    return ac
