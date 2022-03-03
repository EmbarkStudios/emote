from typing import Union, Any, Optional
import logging

import torch
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter

from .callback import Callback


class LoggingCallback(Callback):
    """A Callback that accepts logging calls.

    Logged data is saved on this object and gets written
    by a Logger callback. LoggingCallback therefore doesn't
    care how the data is logged, it only provides a standard
    interface for storing the data to be handled by a Logger."""

    def __init__(self):
        super().__init__()
        self.scalar_logs: dict[str, Union[float, torch.Tensor]] = {}
        self.hist_logs: dict[str, Union[float, torch.Tensor]] = {}

    def log_scalar(self, key: str, value: Union[float, torch.Tensor]):
        """Use log_scalar to periodically log scalar data."""
        if isinstance(value, torch.Tensor):
            self.scalar_logs[key] = value.item()
        else:
            self.scalar_logs[key] = value

    def log_histogram(self, key: str, value: torch.Tensor):
        self.hist_logs[key] = value.detach()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["scalar_logs"] = self.scalar_logs
        state_dict["hist_logs"] = self.hist_logs
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.scalar_logs = state_dict.pop("scalar_logs")
        self.hist_logs = state_dict.pop("hist_logs")
        super().load_state_dict(state_dict)


class LossCallback(LoggingCallback):
    """Losses are callbacks that implement a *loss function*."""

    def __init__(
        self,
        name: str,
        optimizer: optim.Optimizer,
        max_grad_norm: float,
        data_group: str,
    ):
        super().__init__()
        self.data_group = data_group
        self.name = name
        self.optimizer = optimizer
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

        self.log_scalar(f"loss/{self.name}_loss", loss)
        self.log_scalar(f"loss/{self.name}_gradient_norm", grad_norm)

    def state_dict(self):
        state = super().state_dict()
        state["optimizer_state_dict"] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict.pop("optimizer_state_dict"))
        super().load_state_dict(state_dict)

    @Callback.extend
    def loss(self, *args, **kwargs) -> Tensor:
        """The loss method needs to be overwritten to implement a loss.

        :return: A PyTorch tensor of shape (batch,)."""
        raise NotImplementedError


class TensorboardLogger(Callback):
    """Logs the provided loggable callbacks to tensorboard."""

    def __init__(
        self,
        callbacks: list[LoggingCallback],
        writer: SummaryWriter,
        log_interval: int,
        log_by_samples: bool = False,
    ):
        super().__init__(cycle=log_interval)
        self._cbs = callbacks
        self._writer = writer
        self._log_samples = log_by_samples

    def log_scalars(self, step, suffix=None):
        """Logs scalar logs adding optional suffix on the first level.

        **Example:** If k='training/loss' and suffix='bp_step', k will be renamed to
        'training_bp_step/loss'.
        """
        for cb in self._cbs:
            for k, v in cb.scalar_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                self._writer.add_scalar(k, v, step)

    def end_cycle(self, bp_step, bp_samples):
        self.log_scalars(bp_step, suffix="bp_step")
        if self._log_samples:
            self.log_scalars(bp_samples, suffix="bp_samples")


class TerminalLogger(Callback):
    """Logs the provided loggable callbacks to the python logger."""

    def __init__(
        self,
        callbacks: list[LoggingCallback],
        log_interval: int,
    ):
        super().__init__(cycle=log_interval)
        self._cbs = callbacks

    def log_scalars(self, step, suffix=None):
        """Logs scalar logs adding optional suffix on the first level.

        **Example:** If k='training/loss' and suffix='bp_step', k will be renamed to
        'training_bp_step/loss'.
        """
        for cb in self._cbs:
            for k, v in cb.scalar_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                logging.info("%s@%s:\t%.4f", k, step, v)

    def end_cycle(self, bp_step):
        self.log_scalars(bp_step)


class Checkpointer(Callback):
    """Checkpointer writes out a checkpoint every n steps.

    Exactly what is written to the checkpoint is determined by the networks and
    callbacks supplied in the constructor.

    :param networks (list[nn.Module]): A list of networks that should be saved.
    :param callbacks (list[Callback]): A list of callbacks the should be saved.
    :param path (str): The path to where the checkpoint should be stored.
    :param checkpoint_interval (int): Number of backprops between checkpoints.
    :param optimizers (Optional[list[optim.Optimizer]]): Optional list of optimizers
        to save. Usually optimizers are handled by their respective callbacks but
        if you give them to this list they will be handled explicitly.
    """

    def __init__(
        self,
        *,
        networks: list[nn.Module],
        callbacks: list[Callback],
        path: str,
        checkpoint_interval: int,
        optimizers: Optional[list[optim.Optimizer]] = None,
    ):
        super().__init__(cycle=checkpoint_interval)
        self._nets = networks
        self._cbs = callbacks
        self._path = path
        self._checkpoint_index = 0
        self._opts: list[optim.Optimizer] = optimizers if optimizers else []

    def end_cycle(self, inf_step, bp_step, bp_samples):
        state_dict = {}
        state_dict["callback_state_dicts"] = [cb.state_dict() for cb in self._cbs]
        state_dict["network_state_dicts"] = [net.state_dict() for net in self._nets]
        state_dict["optim_state_dicts"] = [opt.state_dict() for opt in self._opts]
        state_dict["training_state"] = {
            "checkpoint_index": self._checkpoint_index,
            "inf_step": inf_step,
            "bp_step": bp_step,
            "bp_samples": bp_samples,
        }
        torch.save(state_dict, f"{self._path}.{self._checkpoint_index}.tar")
        self._checkpoint_index += 1


class CheckpointLoader(Callback):
    """CheckpointLoader loads a checkpoint like the one created by Checkpointer.

    This is intended for resuming training given a specific checkpoint index. If you
    want to do something more specific, like only restore a specific network, it is
    probably easier to just do it explicitly when the network is constructed.

    :param networks (list[nn.Module]): A list of networks that should be saved.
    :param callbacks (list[Callback]): A list of callbacks the should be saved.
    :param path (str): The path to where the checkpoint should be stored.
    :param checkpoint_index (int): Which checkpoint to load.
    :param reset_training_steps (bool): If False, start training at bp_steps=0 etc.
        Otherwise start the training at whatever step and state the checkpoint has
        saved.
    :param optimizers (Optional[list[optim.Optimizer]]): Optional list of optimizers
        to save. Usually optimizers are handled by their respective callbacks but
        if you give them to this list they will be handled explicitly.
    """

    def __init__(
        self,
        networks: list[nn.Module],
        callbacks: list[Callback],
        path: str,
        checkpoint_index: int,
        reset_training_steps: bool = False,
        optimizers: Optional[list[optim.Optimizer]] = None,
    ):
        super().__init__()
        self._nets = networks
        self._cbs = callbacks
        self._path = path
        self._checkpoint_index = checkpoint_index
        self._reset_training_steps = reset_training_steps
        self._opts: list[optim.Optimizer] = optimizers if optimizers else []

    def begin_training(self):
        state_dict: dict = torch.load(f"{self._path}.{self._checkpoint_index}.tar")
        for cb, state in zip(self._cbs, state_dict["callback_state_dicts"]):
            cb.load_state_dict(state)
        for net, state in zip(self._nets, state_dict["network_state_dicts"]):
            net.load_state_dict(state)
        for opt, state in zip(self._opts, state_dict["optim_state_dicts"]):
            opt.load_state_dict(state)
        if self._reset_training_steps:
            return {}
        return state_dict.get("training_state", {})
