import logging
import os
import time
import warnings

from typing import List

import torch

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

        name = f"checkpoint_{self._checkpoint_index}.tar"
        final_path = os.path.join(self._folder_path, name)
        state_dict = {
            "callback_state_dicts": {cb.name: cb.state_dict() for cb in self._cbs},
            "training_state": {
                "latest_checkpoint": final_path,
                "checkpoint_index": self._checkpoint_index,
            },
        }
        torch.save(state_dict, final_path)
        self._checkpoint_index += 1


class CheckpointLoader(Callback):
    """CheckpointLoader loads a checkpoint like the one created by Checkpointer.

    This is intended for resuming training given a specific checkpoint index. It also
    enables you to load network weights, optimizer, or other callback hyper-params
    independently.  If you want to do something more specific, like only restore a
    specific network (outside a callback), it is probably easier to just do it
    explicitly when the network is constructed.

    :param callbacks (List[Callback]): A list of callbacks that should be restored.
    :param run_root (str): The root path to where the run artifacts should be stored.
    :param checkpoint_index (int): Which checkpoint to load.
    :param load_weights (bool): If True, it loads the network weights
    :param load_optimizers (bool): If True, it loads the optimizer state
    :param load_hparams (bool): If True, it loads other callback hyper-params
    :param storage_subdirectory (str): The subdirectory where the checkpoints are
        stored.
    """

    def __init__(
        self,
        *,
        callbacks: List[Callback],
        run_root: str,
        checkpoint_index: int,
        load_weights: bool = True,
        load_optimizers: bool = True,
        load_hparams: bool = True,
        storage_subdirectory: str = "checkpoints",
    ):
        super().__init__()
        self._run_root = run_root
        self._checkpoint_index = checkpoint_index
        self._folder_path = os.path.join(run_root, storage_subdirectory)

        self._load_weights = load_weights
        self._load_optimizers = load_optimizers
        self._load_hparams = load_hparams

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
            cb.load_state_dict(
                state, self._load_weights, self._load_optimizers, self._load_hparams
            )

        return_value = {}
        if self._load_hparams:
            return_value = state_dict.get("training_state", {})
        duration = time.time() - start_time
        logging.info(f"Loaded checkpoint from {final_path} in {duration:.2f}s")
        return return_value


class InvalidCheckpointLocation(ValueError):
    pass
