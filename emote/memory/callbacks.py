import logging
import os
import time

from emote import Callback
from emote.callbacks import LoggingCallback
from emote.utils import BlockTimers

from .table import Table


class MemoryImporterCallback(Callback):
    """Load and validate a previously exported memory"""

    def __init__(
        self,
        target_memory_name: str,
        experiment_load_dir: str,
        load_fname_override=None,
    ):
        super().__init__()
        self._target_memory_name = target_memory_name
        self._load_fname_override = load_fname_override
        self._load_dir = experiment_load_dir

    @Callback.keys_from_member(memory="_target_memory_name")
    def begin_training(self, memory: Table):
        if self._load_fname_override not in (None, ""):
            restore_path = os.path.join(self._load_dir, self._load_fname_override)
        else:
            restore_path = os.path.join(
                self._load_dir, f"{self._target_memory_name}_export"
            )

        if not os.path.exists(restore_path + ".zip"):
            raise FileNotFoundError(
                f"Failed to load memory dump: {restore_path} does not exist."
            )

        memory.restore(restore_path)
        logging.info(f"Loading memory dump {restore_path}")


class MemoryExporterCallback(LoggingCallback):
    """Export the memory at regular intervals"""

    def __init__(
        self,
        target_memory_name,
        inf_steps_per_memory_export,
        min_time_per_export: int = 600,
    ):
        super().__init__()
        assert (
            inf_steps_per_memory_export > 10_000
        ), "exporting a memory is a slow operation and shouldn't be done too often"
        self._target_memory_name = target_memory_name
        self._inf_steps_per_memory_export = inf_steps_per_memory_export
        self._min_time_per_export = min_time_per_export

        self._next_export = inf_steps_per_memory_export
        self._next_export_time = time.monotonic() + min_time_per_export
        self._scopes = BlockTimers()

    def begin_training(self, inf_step: int = 0):
        self._next_export += (
            inf_step  # offset by already existing inf steps, in case resuming
        )

    @Callback.keys_from_member(memory="_target_memory_name")
    def end_batch(
        self,
        inf_step: int = 0,
        experiment_root_path: str = None,
        memory: Table = None,
    ):
        assert experiment_root_path is not None
        assert memory is not None

        has_enough_data = inf_step > self._next_export
        time_now = time.monotonic()
        has_enough_time = time_now > self._next_export_time

        if has_enough_data and has_enough_time:
            export_path = os.path.join(
                experiment_root_path, f"{self._target_memory_name}_export"
            )
            with self._scopes.scope("export"):
                memory.store(export_path)

            self._next_export = inf_step + self._inf_steps_per_memory_export
            self._next_export_time = time_now + self._min_time_per_export

        for name, (mean, var) in self._scopes.stats().items():
            self.log_scalar(
                f"memory/{self._target_memory_name}/{name}/timing/mean", mean
            )
            self.log_scalar(f"memory/{self._target_memory_name}/{name}/timing/var", var)
