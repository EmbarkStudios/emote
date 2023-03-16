# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file contains codes and texts that are copied from
# https://github.com/facebookresearch/mbrl-lib

import pickle

from typing import Union

import numpy as np
import torch

from torch.nn import functional as F

from emote.typing import TensorType


def truncated_linear(
    min_x: float, max_x: float, min_y: float, max_y: float, x: float
) -> float:
    """Truncated linear function.
    Implements the following function:
        f1(x) = min_y + (x - min_x) / (max_x - min_x) * (max_y - min_y)
        f(x) = min(max_y, max(min_y, f1(x)))
    If max_x - min_x < 1e-10, then it behaves as the constant f(x) = max_y
    """
    if max_x - min_x < 1e-10:
        return max_y
    if x <= min_x:
        y: float = min_y
    else:
        dx = (x - min_x) / (max_x - min_x)
        dx = min(dx, 1.0)
        y = dx * (max_y - min_y) + min_y
    return y


def gaussian_nll(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """Negative log-likelihood for Gaussian distribution

    Args:
        pred_mean (tensor): the predicted mean.
        pred_logvar (tensor): the predicted log variance.
        target (tensor): the target value.
        reduce (bool): if ``False`` the loss is returned w/o reducing.
            Defaults to ``True``.

    Returns:
        (tensor): the negative log-likelihood.
    """
    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-pred_logvar).exp()
    losses = l2 * inv_var + pred_logvar
    if reduce:
        return losses.sum(dim=1).mean()
    return losses
