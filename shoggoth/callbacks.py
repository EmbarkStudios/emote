from typing import Dict, List, Union

import torch
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter

from .callback import Callback


class LoggingCallback(Callback):
    """A Callback that accept logging calls.

    Logged data is saved on this object and gets written
    by the Logger callback."""

    def __init__(self):
        super().__init__()
        self.scalar_logs: Dict[str, Union[float, torch.Tensor]] = {}
        self.hist_logs: Dict[str, Union[float, torch.Tensor]] = {}

    def log_scalar(self, key: str, value: Union[float, torch.Tensor]):
        if isinstance(value, torch.Tensor):
            assert value.shape == (1,), "Wrong shape for scalar log tensor."
            self.scalar_logs[key] = value.detach()
        else:
            self.scalar_logs[key] = value

    def log_histogram(self, key: str, value: torch.Tensor):
        self.hist_logs[key] = value.detach()


class LossCallback(LoggingCallback):
    """Losses are callbacks that implement a *loss function*.

    Loss functions must return a Tensor a batch x scalar dimensions."""

    def __init__(
        self,
        name: str,
        optimizer: optim.Optimizer,
        max_grad_norm: float,
    ):
        super().__init__()
        self.name = name
        self.optimizer = optimizer
        self.parameters = [
            p for param_group in optimizer.param_groups for p in param_group["params"]
        ]
        self._max_grad_norm = max_grad_norm

    # def setup_schedules(self, total_timesteps): #TODO(singhblom) implement correct schedule for torch
    #     self._lr_schedule.setup(total_timesteps)

    def backward(self, *args, **kwargs):
        self.optimizer.zero_grad()
        loss = self.loss(*args, **kwargs)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters, self._max_grad_norm)
        self.optimizer.step()

        self.log_scalar(f"loss/{self.name}_loss", loss)
        self.log_scalar(f"loss/{self.name}_gradient_norm", grad_norm)

    def end_batch(self, **kwargs):
        self.log_scalar(
            f"training/{self.name}_learning_rate", self.optimizer.param_groups[0]["lr"]
        )

    def get_state(self):
        return {"optimizer_state_dict": self.optimizer.state_dict()}

    def load_state(self, state):
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

    def loss(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class Logger(Callback):
    def __init__(
        self,
        callbacks: List[LoggingCallback],
        writer: SummaryWriter,
        bp_step: bool = True,
        bp_samples: bool = True,
    ):
        super().__init__()
        self._callbacks = callbacks
        self._writer = writer
        self._step = bp_step
        self._samples = bp_samples

    def log_scalars(self, step, suffix=None):
        """Logs scalar logs adding optional suffix on the first level.
        E.g. If k='training/loss' and suffix='bp_step', k will be renamed to 'training_bp_step/loss'."""
        for cb in self._callbacks:
            for k, v in cb.scalar_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                self._writer.add_scalar(k, v, step)

    def end_batch(self, should_bp_log, bp_step, bp_samples, **kwargs):
        if should_bp_log and self._step:
            self.log_scalars(bp_step, suffix="bp_step")
        if should_bp_log and self._samples:
            self.log_scalars(bp_samples, suffix="bp_samples")


class Checkpointer(Callback):
    def __init__(
        self,
        net: nn.Module,
        callbacks: List[LoggingCallback],
        path: str,
    ):
        super().__init__()
        self._net = net
        self._callbacks = callbacks
        self._path = path

    def end_cycle(self, cycle_index, inf_step, bp_step, bp_samples, **kwargs):
        state_dict = {}
        state_dict["model_state_dict"] = self._net.state_dict()
        state_dict["callbacks"] = [cb.get_state() for cb in self._callbacks]
        state_dict["cycle_index"] = cycle_index
        state_dict["inf_step"] = inf_step
        state_dict["bp_step"] = bp_step
        state_dict["bp_samples"] = bp_samples
        torch.save(state_dict, f"{self._path}.{cycle_index}.tar")


class CheckpointLoader(Callback):
    def __init__(
        self,
        net: nn.Module,
        callbacks: List[LoggingCallback],
        path: str,
        cycle_index: int,
        reset_training_steps: bool = False,
    ):
        super().__init__()
        self._net = net
        self._callbacks = callbacks
        self._path = path
        self._cycle_index = cycle_index
        self._reset_training_steps = reset_training_steps

    def begin_training(self, **kwargs):
        state_dict = torch.load(f"{self._path}.{self._cycle_index}.tar")
        self._net.load_state_dict(state_dict["model_state_dict"])
        for cb, state in zip(self._callbacks, state_dict["callbacks"]):
            cb.load_state(state)
        if self._reset_training_steps:
            return {}
        return {
            "cycle_index": state_dict["cycle_index"],
            "inf_step": state_dict["inf_step"],
            "bp_step": state_dict["bp_step"],
            "bp_samples": state_dict["bp_samples"],
        }
