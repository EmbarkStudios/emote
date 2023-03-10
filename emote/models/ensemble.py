# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence, Union

# This file contains codes and texts that are copied from
# https://github.com/facebookresearch/mbrl-lib
import numpy as np
import torch

from torch import nn as nn
from torch.nn import functional as F

from emote.utils.math import gaussian_nll


def truncated_normal_(
    tensor: torch.Tensor, mean: float = 0, std: float = 1
) -> torch.Tensor:
    """Samples from a truncated normal distribution in-place.

    Args:
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
    """Efficient linear layer for ensemble models."""

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        self.bias = (
            nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            if bias
            else None
        )
        self.elite_models: list[int] = None
        self.use_only_elite = False

    def forward(self, x):
        if self.use_only_elite:
            xw = x.matmul(self.weight[self.elite_models, ...])
            if self.bias is not None:
                return xw + self.bias[self.elite_models, ...]
            else:
                return xw
        else:
            xw = x.matmul(self.weight)
            if self.bias is not None:
                return xw + self.bias
            else:
                return xw

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite


class EnsembleBase(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_members: int,
        device: Union[str, torch.device],
        propagation_method: str = None,
        deterministic: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_members = num_members
        self.propagation_method = propagation_method
        self.propagation_indices: torch.Tensor = None
        self.elite_models: list[int] = None
        self.deterministic = deterministic
        self.device = torch.device(device)
        self.to(device)

    def __len__(self):
        return self.num_members

    def forward(
        self,
        x: torch.Tensor,
        use_propagation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.
        Args:
            x (tensor): the input to the model.
            use_propagation (bool): if False, the propagation method will be ignored

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output.
        """
        raise NotImplementedError("forward method must be implemented.")

    def loss(
        self,
        model_in: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, any]]:
        """Computes loss.
        Args:
            model_in (tensor): input tensor.
            target (tensor): target tensor.
        Returns:
            (a tuple of tensor and dict): a loss tensor and a dict which includes
            extra info.
        """
        raise NotImplementedError("loss method must be implemented.")

    def sample(
        self,
        model_input: torch.Tensor,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Samples an output from the model using .

        Args:
            model_input (tensor): the observation and action at.
                "sample" (e.g., the mean prediction). Defaults to ``False``.
            rng (`torch.Generator`): a random number generator.

        Returns:
            (tuple): predicted observation, rewards, terminal indicator and model
                state dictionary. Everything but the observation is optional, and can
                be returned with value ``None``.
        """
        if self.deterministic:
            return self.forward(model_input)[0]
        means, logvars = self.forward(model_input)
        variances = logvars.exp()
        stds = torch.sqrt(variances)
        assert rng is not None  # rng is required for stochastic inference
        return torch.normal(means, stds, generator=rng)

    def set_elite(self, elite_indices: Sequence[int]):
        self.elite_models = list(elite_indices)

    def save(self, save_dir: str):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, save_dir)

    def load(self, load_dir: str):
        """Loads the model from the given path."""
        model_dict = torch.load(load_dir)
        self.load_state_dict(model_dict["state_dict"])
        self.elite_models = model_dict["elite_models"]


class EnsembleOfGaussian(EnsembleBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        propagation_method: str = "expectation",
        learn_logvar_bounds: bool = False,
    ):
        super().__init__(in_size, out_size, ensemble_size, device, propagation_method)
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

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        mean = mean_and_logvar[..., : self.out_size]
        logvar = mean_and_logvar[..., self.out_size :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def _forward_from_indices(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        mean, logvar = self._default_forward(shuffled_x, only_elite=True)
        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    def _forward_ensemble(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Valid propagation options are:
            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.
        """
        if self.propagation_method is None:
            mean, logvar = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            if self.propagation_indices is None:
                raise ValueError(
                    "When using propagation='fixed_model', `propagation_indices` must be provided."
                )
            return self._forward_from_indices(x, self.propagation_indices)
        if self.propagation_method == "expectation":
            mean, logvar = self._default_forward(x, only_elite=True)
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(
        self,
        x: torch.Tensor,
        use_propagation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_propagation and self.propagation_method is not None:
            return self._forward_ensemble(x)
        return self._default_forward(x)

    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes Gaussian NLL loss."""
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        nll = (
            gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
            .mean((1, 2))  # average over batch and target dimension
            .sum()  # sum over ensemble dimension
        )
        nll += self.logvar_loss_weight * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, any]]:
        return self._nll_loss(model_in, target), {}
