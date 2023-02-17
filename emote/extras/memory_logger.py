"""
Logger that logs the memory consumption and memory consumption growth rate.
"""

import os

import psutil

from emote.callbacks import LoggingCallback


class MemoryLogger(LoggingCallback):
    def __init__(self):
        super().__init__(cycle=1_000)
        self._proc = psutil.Process(os.getpid())
        self._previous_memory = self._proc.memory_info().rss / (1024 * 1024)
        self._previous_bp_step = 0

    def on_cycle_end(self, bp_step, bp_samples):
        memory_now = self._proc.memory_info().rss / (1024 * 1024)
        self.log_scalar("ram/usage_mb", memory_now)

        memory_delta = memory_now - self._previous_memory
        step_delta = bp_step - self._previous_bp_step
        if step_delta > 0:
            self.log_scalar("memory_stats/growth", memory_delta / step_delta)

        self._previous_memory = memory_now
        self._previous_bp_step = bp_step
