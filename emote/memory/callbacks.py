import logging
import os
import warnings

from emote import Callback
from emote.callbacks import LoggingMixin
from emote.utils import BlockTimers

from .table import Table


class MemoryImporterCallback(Callback):
    """Load and validate a previously exported memory"""

    def __init__(
        self,
        memory: Table,
        target_memory_name: str,
        experiment_load_dir: str,
        load_fname_override=None,
    ):
        super().__init__()
        self._order = (
            -1
        )  # this is to ensure that this callback is called before the others
        self.memory = memory
        self._target_memory_name = target_memory_name
        self._load_fname_override = load_fname_override
        self._load_dir = experiment_load_dir

    def begin_training(self):
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

        self.memory.restore(restore_path)
        logging.info(f"Loading memory dump {restore_path}")


class MemoryExporterCallback(LoggingMixin, Callback):
    """Export the memory at regular intervals"""

    def __init__(
        self,
        memory: Table,
        target_memory_name,
        inf_steps_per_memory_export,
        experiment_root_path: str,
    ):
        rec_min_inf_steps = 10_000
        if inf_steps_per_memory_export < rec_min_inf_steps:
            warnings.warn(
                f"Exporting a memory is a slow operation "
                f"and should not be done too often. "
                f"Current inf_step is {inf_steps_per_memory_export}, "
                f"while the recommended minimum is {rec_min_inf_steps}.",
                UserWarning,
            )

        super().__init__(cycle=inf_steps_per_memory_export)
        self.memory = memory
        self.experiment_root_path = experiment_root_path
        self._target_memory_name = target_memory_name
        self._inf_steps_per_memory_export = inf_steps_per_memory_export
        self._scopes = BlockTimers()

    def end_cycle(self):
        export_path = os.path.join(
            self.experiment_root_path, f"{self._target_memory_name}_export"
        )
        with self._scopes.scope("export"):
            self.memory.store(export_path)

        for name, (mean, var) in self._scopes.stats().items():
            self.log_scalar(
                f"memory/{self._target_memory_name}/{name}/timing/mean", mean
            )
            self.log_scalar(f"memory/{self._target_memory_name}/{name}/timing/var", var)
