"""
Logger that logs the memory consumption and memory consumption growth rate.
"""

import os

import psutil

from emote.callback import Callback
from emote.callbacks import LoggingMixin


class SystemLogger(LoggingMixin, Callback):
    def __init__(self):
        super().__init__(cycle=1_000)
        self._proc = psutil.Process(os.getpid())
        self._previous_memory = self._proc.memory_info().rss / (1024 * 1024)
        self._previous_bp_step = 0

    def end_cycle(self, bp_step, bp_samples):
        memory_now = self._proc.memory_info().rss / (1024 * 1024)
        self.log_scalar("system/ram_usage_mb", memory_now)

        memory_delta = memory_now - self._previous_memory
        step_delta = bp_step - self._previous_bp_step
        if step_delta > 0:
            self.log_scalar(
                "system/ram_usage_growth_mb_step", memory_delta / step_delta
            )

        self._previous_memory = memory_now
        self._previous_bp_step = bp_step

        # unix-style summed load
        cpu_percent = self._proc.cpu_percent()
        self.log_scalar("system/cpu_load", cpu_percent)

        # Requires torch 1.11
        # import torch
        # if torch.cuda.is_available() and torch.cuda.is_initialized():
        #     gpu_load = torch.cuda.utilization()
        #     self.log_scalar("system/gpu_load", gpu_load)
