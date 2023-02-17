# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file contains codes/text mostly restructured from the following github repository
# https://github.com/facebookresearch/mbrl-lib

import torch
import numpy as np
from torch.nn import functional as F
from emote.typing import TensorType
from typing import Union
import pickle


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


class Normalizer:
    """Class that keeps a running mean and variance and normalizes data accordingly.

    The statistics kept are stored in torch tensors.

    Args:
        in_size (int): the size of the data that will be normalized.
        device (torch.device): the device in which the data will reside.
        dtype (torch.dtype): the data type to use for the normalizer.
    """

    _STATS_FNAME = "env_stats.pickle"

    def __init__(self, in_size: int, device: torch.device, dtype=torch.float32):
        self.mean = torch.zeros((1, in_size), device=device, dtype=dtype)
        self.std = torch.ones((1, in_size), device=device, dtype=dtype)
        self.eps = 1e-12 if dtype == torch.double else 1e-5
        self.device = device

    def update_stats(self, data: TensorType):
        """Updates the stored statistics using the given data.

        Equivalent to `self.stats.mean = data.mean(0) and self.stats.std = data.std(0)`.

        Args:
            data (np.ndarray or torch.Tensor): The data used to compute the statistics.
        """
        assert data.ndim == 2 and data.shape[1] == self.mean.shape[1]
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device)
        self.mean = data.mean(0, keepdim=True)
        self.std = data.std(0, keepdim=True)
        self.std[self.std < self.eps] = 1.0

    def normalize(self, val: Union[float, TensorType]) -> torch.Tensor:
        """Normalizes the value according to the stored statistics.

        Equivalent to (val - mu) / sigma, where mu and sigma are the stored mean and
        standard deviation, respectively.

        Args:
            val (float, np.ndarray or torch.Tensor): The value to normalize.

        Returns:
            (torch.Tensor): The normalized value.
        """
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        return (val - self.mean) / self.std

    def denormalize(self, val: Union[float, TensorType]) -> torch.Tensor:
        """De-normalizes the value according to the stored statistics.

        Equivalent to sigma * val + mu, where mu and sigma are the stored mean and
        standard deviation, respectively.

        Args:
            val (float, np.ndarray or torch.Tensor): The value to de-normalize.

        Returns:
            (torch.Tensor): The de-normalized value.
        """
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        return self.std * val + self.mean

    def load(self, results_dir: str):
        """Loads saved statistics from the given path."""
        with open(results_dir, "rb") as f:
            stats = pickle.load(f)
            self.mean = torch.from_numpy(stats["mean"]).to(self.device)
            self.std = torch.from_numpy(stats["std"]).to(self.device)

    def save(self, save_dir: str):
        """Saves stored statistics to the given path."""
        with open(save_dir, "wb") as f:
            pickle.dump(
                {"mean": self.mean.cpu().numpy(), "std": self.std.cpu().numpy()}, f
            )

