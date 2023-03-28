# This file contains codes and texts that are copied from
# https://github.com/facebookresearch/mbrl-lib
from typing import Optional, Union

import numpy as np
import torch

from torch import nn as nn
from torch.nn import GaussianNLLLoss, functional as F


def truncated_normal_(
    tensor: torch.Tensor, mean: float = 0, std: float = 1
) -> torch.Tensor:
    """Samples from a truncated normal distribution in-place.

    Arguments:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        bound_violations = torch.sum(cond).item()
        if bound_violations == 0:
            break
        tensor[cond] = torch.normal(
            mean, std, size=(bound_violations,), device=tensor.device
        )
    return tensor


def truncated_normal_init(m: nn.Module):
    """Initializes the weights of the given module using a truncated normal distribution."""
    if isinstance(m, nn.Linear):
        input_dim = m.weight.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        truncated_normal_(m.weight.data, std=stddev)
        m.bias.data.fill_(0.0)
    if isinstance(m, EnsembleLinearLayer):
        num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_members):
            truncated_normal_(m.weight.data[i], std=stddev)
        m.bias.data.fill_(0.0)


class EnsembleLinearLayer(nn.Module):
    """Linear layer for ensemble models.

    Arguments:
        num_members (int): the ensemble size
        in_size (int): the input size of the model
        out_size (int): the output size of the model
    """

    def __init__(self, num_members: int, in_size: int, out_size: int):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))

    def forward(self, x):
        return x.matmul(self.weight) + self.bias


class EnsembleOfGaussian(nn.Module):
    def __init__(
        self,
        *,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 256,
        learn_logvar_bounds: bool = False,
        deterministic: bool = False,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_members = ensemble_size
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.nll_loss = GaussianNLLLoss(reduction="none")

        activation_func = nn.ReLU()

        hidden_layers = [
            nn.Sequential(
                EnsembleLinearLayer(ensemble_size, in_size, hid_size), activation_func
            )
        ]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    EnsembleLinearLayer(ensemble_size, hid_size, hid_size),
                    activation_func,
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.mean_and_logvar = EnsembleLinearLayer(
            ensemble_size, hid_size, 2 * out_size
        )
        self.min_logvar = nn.Parameter(
            -10 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
        )
        self.max_logvar = nn.Parameter(
            0.5 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
        )
        self.logvar_loss_weight = 0.01
        self.apply(truncated_normal_init)
        self.to(self.device)

    def default_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        mean = mean_and_logvar[..., : self.out_size]
        logvar = mean_and_logvar[..., self.out_size :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        Arguments:
            x (tensor): the input to the model.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output.
        """
        assert x.ndim == 2
        x = x.unsqueeze(0)
        mean, logvar = self.default_forward(x)
        return mean.mean(dim=0), logvar.mean(dim=0)

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, any]]:
        """Computes Gaussian NLL loss.

        Arguments:
            model_in (tensor): input tensor.
            target (tensor): target tensor.

        Returns:
            (a tuple of tensor and dict): a loss tensor and a dict which includes
            extra info.
        """
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar = self.default_forward(model_in)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        nll = (
            self.nll_loss(pred_mean, target, torch.exp(pred_logvar))
            .mean((1, 2))  # average over batch and target dimension
            .sum()  # sum over ensemble dimension
        )

        nll += self.logvar_loss_weight * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll, {}

    def sample(
        self,
        model_input: torch.Tensor,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Samples next observation, reward and terminal from the model using the ensemble.

        Args:
            model_input (tensor): the observation and action.
            rng (torch.Generator): a random number generator.

        Returns:
            (tuple): predicted observation, rewards, terminal indicator and model
                state dictionary.
        """
        if self.deterministic:
            return self.forward(model_input)[0]
        means, logvars = self.forward(model_input)
        variances = logvars.exp()
        stds = torch.sqrt(variances)
        return torch.normal(means, stds, generator=rng)

    def save(self, save_dir: str):
        """Saves the model to the given directory."""
        model_dict = {"state_dict": self.state_dict()}
        torch.save(model_dict, save_dir)

    def load(self, load_dir: str):
        """Loads the model from the given path."""
        model_dict = torch.load(load_dir)
        self.load_state_dict(model_dict["state_dict"])
