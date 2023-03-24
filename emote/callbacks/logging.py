import logging
import time

from collections import deque
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from emote.callback import Callback


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

        try:
            import wandb
        except ImportError as root:
            raise ImportError(
                "enable the optional `wandb` future to use the WBLogger"
            ) from root

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
