import logging
import os

from emote.callback import Callback
from emote.memory.table import Table


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
        self._order = -1  # this is to ensure that this callback is called before the others
        self.memory = memory
        self._target_memory_name = target_memory_name
        self._load_fname_override = load_fname_override
        self._load_dir = experiment_load_dir

    def begin_training(self):
        if self._load_fname_override not in (None, ""):
            restore_path = os.path.join(self._load_dir, self._load_fname_override)
        else:
            restore_path = os.path.join(self._load_dir, f"{self._target_memory_name}_export")

        if not os.path.exists(restore_path + ".zip"):
            raise FileNotFoundError(f"Failed to load memory dump: {restore_path} does not exist.")

        self.memory.restore(restore_path)
        logging.info(f"Loading memory dump {restore_path}")
