from typing import Any, Dict, Optional

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

    def state_dict(self):
        state = super().state_dict()
        if self.optimizer:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.network:
            state["network_state_dict"] = self.network.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.optimizer:
            self.optimizer.load_state_dict(state_dict.pop("optimizer_state_dict"))
        if self.network:
            self.network.load_state_dict(state_dict.pop("network_state_dict"))
        super().load_state_dict(state_dict)

    @Callback.extend
    def loss(self, *args, **kwargs) -> Tensor:
        """The loss method needs to be overwritten to implement a loss.

        :return: A PyTorch tensor of shape (batch,)."""
        raise NotImplementedError
