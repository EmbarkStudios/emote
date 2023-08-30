import logging
import time

from typing import List

from torch.utils.tensorboard import SummaryWriter

from emote.callback import Callback
from emote.mixins.logging import LoggingMixin


class TensorboardLogger(Callback):
    """Logs the provided loggable callbacks to tensorboard."""

    def __init__(
        self,
        loggables: List[LoggingMixin],
        writer: SummaryWriter,
        log_interval: int,
        log_by_samples: bool = False,
    ):
        super().__init__(cycle=log_interval)
        self._logs = loggables
        self._writer = writer
        self._log_samples = log_by_samples

    def begin_training(self):
        self._start_time = time.monotonic()

    def end_cycle(self, bp_step, bp_samples):
        suffix = "bp_step"

        for cb in self._logs:
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

            for k, v in cb.hist_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                self._writer.add_histogram(k, v, bp_step)

        time_since_start = time.monotonic() - self._start_time
        self._writer.add_scalar(
            "performance/bp_samples_per_sec", bp_samples / time_since_start, bp_step
        )
        self._writer.add_scalar(
            "performance/bp_steps_per_sec", bp_step / time_since_start, bp_step
        )

        self._writer.flush()


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
        print("Step %s" % step)
        for log in self._logs:
            for k, v in log.scalar_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                padding = " " * (30 - len(k))
                print(f"\t{k[:30]}{padding}: {v:.4f}")

    def end_cycle(self, bp_step):
        self.log_scalars(bp_step)
