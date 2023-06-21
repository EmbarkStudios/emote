import logging
import os
import time
import warnings

from typing import List, Optional

import torch

from torch import nn, optim

from emote.callback import Callback


class Checkpointer(Callback):
    """Checkpointer writes out a checkpoint every n steps.

    Exactly what is written to the checkpoint is determined by the callbacks
    supplied in the constructor.

    :param callbacks (List[Callback]): A list of callbacks that should be saved.
    :param run_root (str): The root path to where the run artifacts should be stored.
    :param checkpoint_interval (int): Number of backprops between checkpoints.
    :param storage_subdirectory (str): The subdirectory where the checkpoints are
        stored.
    """

    def __init__(
        self,
        *,
        callbacks: List[Callback],
        run_root: str,
        checkpoint_interval: int,
        checkpoint_index: int = 0,
        storage_subdirectory: str = "checkpoints",
    ):
        super().__init__(cycle=checkpoint_interval)
        self._run_root = run_root
        self._checkpoint_index = checkpoint_index
        self._folder_path = os.path.join(run_root, storage_subdirectory)

        self._cbs = []
        names = []
        for cb in callbacks:
            if hasattr(cb, "name"):
                self._cbs.append(cb)
                names.append(cb.name)
            else:
                warnings.warn(
                    f"Checkpointer ignored {cb} because of not "
                    f"having the 'name' field.",
                    UserWarning,
                )

        if len(names) != len(set(names)):
            raise ValueError(
                "Checkpointer is given a list of callbacks with at least"
                "two callbacks with identical names"
            )

    def begin_training(self):
        os.makedirs(self._folder_path, exist_ok=True)

    def end_cycle(self):

        state_dict = {
            "callback_state_dicts": {cb.name: cb.state_dict() for cb in self._cbs},
            "training_state": {
                "checkpoint_index": self._checkpoint_index,
            },
        }
        name = f"checkpoint_{self._checkpoint_index}.tar"
        final_path = os.path.join(self._folder_path, name)
        torch.save(state_dict, final_path)
        self._checkpoint_index += 1


class CheckpointLoader(Callback):
    """CheckpointLoader loads a checkpoint like the one created by Checkpointer.

    This is intended for resuming training given a specific checkpoint index. It is
    also possible to only load neural networks.  If you want to do something more
    specific, like only restore a specific network, it is probably easier to just
    do it explicitly when the network is constructed.

    :param callbacks (List[Callback]): A list of callbacks that should be restored.
    :param run_root (str): The root path to where the run artifacts should be stored.
    :param checkpoint_index (int): Which checkpoint to load.
    :param reset_training_steps (bool): If True, start training at bp_steps=0 etc.
        Otherwise start the training at whatever step and state the checkpoint has
        saved.
    :param only_load_networks (bool): If True, only loads the neural network params
        inside the callback and skips the rest of them.
    :param storage_subdirectory (str): The subdirectory where the checkpoints are
        stored.
    """

    def __init__(
        self,
        *,
        callbacks: List[Callback],
        run_root: str,
        checkpoint_index: int,
        reset_training_steps: bool = False,
        only_load_networks: bool = False,
        storage_subdirectory: str = "checkpoints",
    ):
        super().__init__()
        self._run_root = run_root
        self._checkpoint_index = checkpoint_index
        self._reset_training_steps = reset_training_steps
        self._folder_path = os.path.join(run_root, storage_subdirectory)

        self._only_load_networks = only_load_networks
        self._cbs = []
        names = []
        for cb in callbacks:
            if hasattr(cb, "name"):
                self._cbs.append(cb)
                names.append(cb.name)
            else:
                warnings.warn(
                    f"CheckpointLoader ignored {cb} because of not "
                    f"having the 'name' field.",
                    UserWarning,
                )

        if len(names) != len(set(names)):
            raise ValueError(
                "CheckpointLoader is given a list of callbacks with at least"
                "two callbacks with identical names"
            )

    def begin_training(self):
        start_time = time.time()
        if not os.path.exists(self._folder_path):
            raise InvalidCheckpointLocation(
                f"Checkpoint folder {self._folder_path} was specified but does not exist."
            )
        name = f"checkpoint_{self._checkpoint_index}.tar"
        final_path = os.path.join(self._folder_path, name)
        logging.info(f"Loading checkpoints from {self._folder_path}")
        state_dict: dict = torch.load(final_path)

        for cb in self._cbs:
            state = state_dict["callback_state_dicts"][cb.name]
            cb.load_state_dict(state, self._only_load_networks)

        return_value = {}
        if not self._reset_training_steps:
            return_value = state_dict.get("training_state", {})
        duration = time.time() - start_time
        logging.info(f"Loaded checkpoint from {final_path} in {duration:.2f}s")
        return return_value


class InvalidCheckpointLocation(ValueError):
    pass
