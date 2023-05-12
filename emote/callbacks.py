import logging
import time

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from torch import Tensor, nn, optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from .callback import Callback
from .trainer import TrainingShutdownException


class LoggingMixin:
    """A Mixin that accepts logging calls.

    Logged data is saved on this object and gets written by a
    Logger. This therefore doesn't care how the data is logged, it
    only provides a standard interface for storing the data to be
    handled by a Logger.
    """

    def __init__(self, *, default_window_length: int = 250, **kwargs):
        super().__init__(**kwargs)

        self.scalar_logs: Dict[str, Union[float, torch.Tensor]] = {}
        self.windowed_scalar: Dict[str, deque[Union[float, torch.Tensor]]] = {}
        self.windowed_scalar_cumulative: Dict[str, int] = {}
        self.image_logs: Dict[str, torch.Tensor] = {}
        self.hist_logs: Dict[str, Union[float, torch.Tensor]] = {}
        self.video_logs: Dict[str, Tuple[np.ndarray, int]] = {}

        self._default_window_length = default_window_length

    def log_scalar(self, key: str, value: Union[float, torch.Tensor]):
        """Use log_scalar to periodically log scalar data."""
        if isinstance(value, torch.Tensor):
            self.scalar_logs[key] = value.item()
        else:
            self.scalar_logs[key] = value

    def log_windowed_scalar(self, key: str, value: Union[float, torch.Tensor]):
        """Log scalars using a moving window average.

        By default this will use `default_window_length` from the constructor as the window
        length. It can also be overridden on a per-key basis using the format
        windowed[LENGTH]:foo/bar. Note that this cannot be changed between multiple invocations -
        whichever length is found first will be permanent.
        """

        if key not in self.windowed_scalar:
            # we allow windowed[100]:some_key/foobar to override window size
            if "windowed[" in key:
                p, k = key.split(":")
                length = int(key.split("[")[1][:-1])
                key = k
            else:
                length = self._default_window_length

            self.windowed_scalar[key] = deque(maxlen=length)
            self.windowed_scalar_cumulative[key] = 0

        if isinstance(value, torch.Tensor):
            self.windowed_scalar[key].append(value.item())
        else:
            self.windowed_scalar[key].append(value)

    def log_image(self, key: str, value: torch.Tensor):
        """Use log_image to periodically log image data."""
        if len(value.shape) == 3:
            self.image_logs[key] = value.detach()

    def log_video(self, key: str, value: Tuple[np.ndarray, int]):
        """Use log_scalar to periodically log scalar data."""
        self.video_logs[key] = value

    def log_histogram(self, key: str, value: torch.Tensor):
        self.hist_logs[key] = value.detach()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["scalar_logs"] = self.scalar_logs
        state_dict["hist_logs"] = self.hist_logs
        state_dict["image_logs"] = self.image_logs
        state_dict["video_logs"] = self.video_logs
        state_dict["windowed_scalar"] = {
            k: (list(v), v.maxlen) for (k, v) in self.windowed_scalar.items()
        }
        state_dict["windowed_scalar_cumulative"] = self.windowed_scalar_cumulative
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.scalar_logs = state_dict.pop("scalar_logs")
        self.hist_logs = state_dict.pop("hist_logs")
        self.video_logs = state_dict.pop("video_logs")
        self.image_logs = state_dict.pop("image_logs")
        self.windowed_scalar = {
            k: deque(v[0], maxlen=v[1]) for (k, v) in self.windowed_scalar.items()
        }
        self.windowed_scalar_cumulative = state_dict.pop("windowed_scalar_cumulative")

        super().load_state_dict(state_dict)


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


class TensorboardLogger(Callback):
    """Logs the provided loggable callbacks to tensorboard."""

    def __init__(
        self,
        callbacks: List[LoggingMixin],
        writer: SummaryWriter,
        log_interval: int,
        log_by_samples: bool = False,
    ):
        super().__init__(cycle=log_interval)
        self._cbs = callbacks
        self._writer = writer
        self._log_samples = log_by_samples

    def begin_training(self):
        self._start_time = time.monotonic()

    def end_cycle(self, bp_step, bp_samples):
        suffix = "bp_step"

        for cb in self._cbs:
            for k, v in cb.scalar_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                self._writer.add_scalar(k, v, bp_step)

            for k, v in cb.windowed_scalar.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                self._writer.add_scalar(k, sum(v) / len(v), bp_step)

            for k, v in cb.windowed_scalar_cumulative.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                self._writer.add_scalar(f"{k}/cumulative", v, bp_step)

            for k, v in cb.image_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                self._writer.add_image(k, v, bp_step, dataformats="HWC")

            for k, (video_array, fps) in cb.video_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                self._writer.add_video(k, video_array, bp_step, fps=fps, walltime=None)

        time_since_start = time.monotonic() - self._start_time
        self._writer.add_scalar(
            "performance/bp_samples_per_sec", bp_samples / time_since_start, bp_step
        )
        self._writer.add_scalar(
            "performance/bp_steps_per_sec", bp_step / time_since_start, bp_step
        )


class WBLogger(Callback):
    """Logs the provided loggable callbacks to Weights&Biases."""

    def __init__(
        self,
        callbacks: List[LoggingMixin],
        config: Dict,
        log_interval: int,
    ):
        super().__init__(cycle=log_interval)

        import wandb

        self._cbs = callbacks
        self._config = config

        assert wandb.run is None
        wandb.init(
            project=self._config["wandb_project"],
            name=self._config["wandb_run"],
            config=wandb.helper.parse_config(
                self._config, exclude=("wandb_project", "wandb_run")
            ),
        )

    def begin_training(self):
        self._start_time = time.monotonic()

    def end_cycle(self, bp_step, bp_samples):
        import wandb

        log_dict = {}
        suffix = "bp_step"

        for cb in self._cbs:
            for k, v in cb.scalar_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                log_dict[k] = v

            for k, v in cb.windowed_scalar.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                log_dict[k] = sum(v) / len(v)

            for k, v in cb.windowed_scalar_cumulative.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                log_dict[f"{k}/cumulative"] = v

            for k, v in cb.image_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                log_dict[k] = wandb.Image(v)

            for k, (video_array, fps) in cb.video_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                log_dict[k] = wandb.Video(video_array, fps=fps)

        time_since_start = time.monotonic() - self._start_time
        log_dict["performance/bp_samples_per_sec"] = bp_samples / time_since_start
        log_dict["performance/bp_steps_per_sec"] = bp_step / time_since_start
        log_dict["log/bp_step"] = bp_step
        wandb.log(log_dict)

    def end_training(self):
        import wandb

        wandb.finish()
        return super().end_training()


class TerminalLogger(Callback):
    """Logs the provided loggable callbacks to the python logger."""

    def __init__(
        self,
        callbacks: List[LoggingMixin],
        log_interval: int,
    ):
        super().__init__(cycle=log_interval)
        self._logs = callbacks

    def log_scalars(self, step, suffix=None):
        """Logs scalar logs adding optional suffix on the first level.

        **Example:** If k='training/loss' and suffix='bp_step', k will be renamed to
        'training_bp_step/loss'.
        """
        for log in self._logs:
            for k, v in log.scalar_logs.items():
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

    :param callbacks (List[Callback]): A list of callbacks the should be saved.
    :param path (str): The path to where the checkpoint should be stored.
    :param checkpoint_interval (int): Number of backprops between checkpoints.
    :param optimizers (Optional[List[optim.Optimizer]]): Optional list of optimizers
        to save. Usually optimizers are handled by their respective callbacks but
        if you give them to this list they will be handled explicitly.
    :param networks (Optional[List[nn.Module]]): An optional list of networks that
        should be saved. Usually networks and optimizers are both restored by the
        callbacks which handles their parameters.
    """

    def __init__(
        self,
        *,
        callbacks: List[Callback],
        path: str,
        checkpoint_interval: int,
        checkpoint_index: int = 0,
        optimizers: Optional[List[optim.Optimizer]] = None,
        networks: Optional[List[nn.Module]] = None,
    ):
        super().__init__(cycle=checkpoint_interval)
        self._cbs = callbacks
        self._path = path
        self._checkpoint_index = checkpoint_index
        self._opts: List[optim.Optimizer] = optimizers if optimizers else []
        self._nets: List[nn.Module] = networks if networks else []

    def end_cycle(self):
        state_dict = {}
        state_dict["callback_state_dicts"] = [cb.state_dict() for cb in self._cbs]
        state_dict["network_state_dicts"] = [net.state_dict() for net in self._nets]
        state_dict["optim_state_dicts"] = [opt.state_dict() for opt in self._opts]
        state_dict["training_state"] = {
            "checkpoint_index": self._checkpoint_index,
        }
        torch.save(state_dict, f"{self._path}.{self._checkpoint_index}.tar")
        self._checkpoint_index += 1


class CheckpointLoader(Callback):
    """CheckpointLoader loads a checkpoint like the one created by Checkpointer.

    This is intended for resuming training given a specific checkpoint index. If you
    want to do something more specific, like only restore a specific network, it is
    probably easier to just do it explicitly when the network is constructed.

    :param callbacks (List[Callback]): A list of callbacks the should be restored.
    :param path (str): The path to where the checkpoint should be stored.
    :param checkpoint_index (int): Which checkpoint to load.
    :param reset_training_steps (bool): If False, start training at bp_steps=0 etc.
        Otherwise start the training at whatever step and state the checkpoint has
        saved.
    :param optimizers (Optional[List[optim.Optimizer]]): Optional list of optimizers
        to restore. Usually optimizers are handled by their respective callbacks but
        if you give them to this list they will be handled explicitly.
    :param networks (Optional[List[nn.Module]]): An optional list of networks that
        should be restored. Usually networks and optimizers are both restored by the
        callbacks which handles their parameters.
    """

    def __init__(
        self,
        *,
        callbacks: List[Callback],
        path: str,
        checkpoint_index: int,
        reset_training_steps: bool = False,
        optimizers: Optional[List[optim.Optimizer]] = None,
        networks: Optional[List[nn.Module]] = None,
    ):
        super().__init__()
        self._cbs = callbacks
        self._path = path
        self._checkpoint_index = checkpoint_index
        self._reset_training_steps = reset_training_steps
        self._opts: List[optim.Optimizer] = optimizers if optimizers else []
        self._nets: List[nn.Module] = networks if networks else []

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


class BackPropStepsTerminator(Callback):
    """Terminates training after a given number of backprops.

    :param bp_steps (int): The total number of backprops that the trainer should run
        for.
    """

    def __init__(self, bp_steps: int):
        assert bp_steps > 0, "Training steps must be above 0."
        super().__init__(cycle=bp_steps)

    def end_cycle(self):
        raise TrainingShutdownException()


class FinalLossTestCheck(Callback):
    """Logs the provided loggable callbacks to the python logger."""

    def __init__(
        self,
        callbacks: List[LossCallback],
        cutoffs: List[float],
        test_length: int,
    ):
        super().__init__(cycle=test_length)
        self._cbs = callbacks
        self._cutoffs = cutoffs

    def end_cycle(self):
        for cb, cutoff in zip(self._cbs, self._cutoffs):
            loss = cb.scalar_logs[f"loss/{cb.name}_loss"]
            if loss > cutoff:
                raise Exception(f"Loss for {cb.name} too high: {loss}")
        raise TrainingShutdownException()


class FinalRewardTestCheck(Callback):
    def __init__(
        self,
        callback: LoggingMixin,
        cutoff: float,
        test_length: int,
    ):
        super().__init__(cycle=test_length)
        self._cb = callback
        self._cutoff = cutoff

    def end_cycle(self):
        reward = self._cb.scalar_logs["training/scaled_reward"]
        if reward < self._cutoff:
            raise Exception(f"Reward too low: {reward}")
        raise TrainingShutdownException()


class BatchCallback(LoggingMixin, Callback):
    def __init__(self):
        super().__init__()

    @Callback.extend
    def get_batch(self, *args, **kwargs):
        pass
