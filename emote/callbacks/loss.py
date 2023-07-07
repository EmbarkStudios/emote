from typing import Any, Dict, Optional

import torch

from torch import Tensor, nn, optim
from torch.optim import lr_scheduler

from emote.callback import Callback
from emote.mixins.logging import LoggingMixin


class LossCallback(LoggingMixin, Callback):
    """Losses are callbacks that implement a *loss function*."""

    def __init__(
        self,
        lr_schedule: Optional[optim.lr_scheduler._LRScheduler] = None,
        *,
        name: str,
        network: Optional[nn.Module],
        optimizer: Optional[optim.Optimizer],
        max_grad_norm: float,
        data_group: str,
        log_per_param_weights=False,
        log_per_param_grads=False,
    ):
        super().__init__()
        self.data_group = data_group
        self.name = name
        self.network = network
        self.optimizer = optimizer
        if lr_schedule is None:
            lr_schedule = lr_scheduler.ConstantLR(optimizer, factor=1.0)
        self.lr_schedule = lr_schedule
        self.parameters = [
            p for param_group in optimizer.param_groups for p in param_group["params"]
        ]
        self._max_grad_norm = max_grad_norm
        self._log_per_param_weights = log_per_param_weights
        self._log_per_param_grads = log_per_param_grads
        # Cache parameters and parameter name for all parameters that we optimize.
        # We can use this when debugging per param values and gradients.
        self._named_parameters = (
            {
                n: p
                for n, p in network.named_parameters(recurse=True)
                if any(p is p_ for p_ in self.parameters)
            }
            if self.network is not None
            else {}
        )

    def backward(self, *args, **kwargs):
        self.optimizer.zero_grad()
        loss = self.loss(*args, **kwargs)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters, self._max_grad_norm)
        self.optimizer.step()
        self.lr_schedule.step()
        self.log_scalar(f"loss/{self.name}_lr", self.lr_schedule.get_last_lr()[0])
        self.log_scalar(f"loss/{self.name}_loss", loss)
        self.log_scalar(f"loss/{self.name}_gradient_norm", grad_norm)

        if self._log_per_param_weights or self._log_per_param_grads:
            self.log_per_param_weights_and_grads()

    def log_per_param_weights_and_grads(self):
        def _friendly_shape_str(shape):
            return str(list(shape)).replace("[", "").replace("]", "").replace(", ", "_")

        for name, parameter in self._named_parameters.items():
            split = name.split(".")
            log_name = self.name + "_" + "_".join(split[:-1])
            param_type = split[-1]

            if self._log_per_param_grads and parameter.grad is not None:
                g_shape = _friendly_shape_str(parameter.grad.shape)
                self.log_histogram(
                    f"{param_type}_grads/{log_name}_{g_shape}", parameter.grad
                )
                self.log_scalar(
                    f"{param_type}_grads_l2/{log_name}_{g_shape}",
                    torch.norm(parameter.grad, p=2),
                )

            if self._log_per_param_weights:
                p_shape = _friendly_shape_str(parameter.shape)
                self.log_histogram(f"{param_type}/{log_name}_{p_shape}", parameter)
                self.log_scalar(
                    f"{param_type}_l2/{log_name}_{p_shape}", torch.norm(parameter, p=2)
                )

    def state_dict(self):
        state = super().state_dict()
        if self.optimizer:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.network:
            state["network_state_dict"] = self.network.state_dict()
        return state

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        load_weights: bool = True,
        load_optimizers: bool = True,
        load_hparams: bool = True,
    ):
        if self.network and load_weights:
            self.network.load_state_dict(state_dict.pop("network_state_dict"))

        if self.optimizer and load_optimizers:
            self.optimizer.load_state_dict(state_dict.pop("optimizer_state_dict"))

        super().load_state_dict(state_dict, load_weights, load_optimizers, load_hparams)

    @Callback.extend
    def loss(self, *args, **kwargs) -> Tensor:
        """The loss method needs to be overwritten to implement a loss.

        :return: A PyTorch tensor of shape (batch,)."""
        raise NotImplementedError
