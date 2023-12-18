# Taken from TD-MPC2 repo
# https://github.com/nicklashansen/tdmpc2

import torch
import torch.nn.functional as F

from torch import nn


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


def symlog(x):
    """
    Symmetric logarithmic function.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """
    Symmetric exponential function.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def soft_ce(pred, target, vmin, vmax, num_bins):
    """
    Cross entropy loss between predictions and soft targets.
    """
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, vmin, vmax, num_bins)
    return -(target * pred).sum(-1, keepdim=True)


def two_hot(x, vmin, vmax, num_bins):
    """
    Converts a batch of scalars to soft two-hot encoded targets for discrete regression.
    """
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symlog(x)
    bin_size = (vmax - vmin) / (num_bins - 1)
    x = torch.clamp(symlog(x), vmin, vmax).squeeze(1)
    bin_idx = torch.floor((x - vmin) / bin_size).long()
    bin_offset = ((x - vmin) / bin_size - bin_idx.float()).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.size(0), num_bins, device=x.device)
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x, vmin, vmax, num_bins):
    """
    Converts a batch of soft two-hot encoded vectors to scalars.
    """
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symexp(x)
    bins = torch.linspace(vmin, vmax, num_bins, device=x.device)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * bins, dim=-1, keepdim=True)
    return symexp(x)
