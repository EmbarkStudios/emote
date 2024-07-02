import logging
import os
import time

from typing import Any, Protocol

import torch

from emote.callback import Callback


class Restoree(Protocol):
    name: str

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        load_network: bool = True,
        load_optimizer: bool = True,
        load_hparams: bool = True,
    ):
        ...


class Checkpointer(Callback):
    """Checkpointer writes out a checkpoint every n steps.

    Exactly what is written to the checkpoint is determined by the
    restorees supplied in the constructor.

    :param restorees (list[Restoree]): A list of restorees that should
    be saved.
    :param run_root (str): The root path to where the run artifacts
        should be stored.
    :param checkpoint_interval (int): Number of backprops between
        checkpoints.
    :param storage_subdirectory (str): The subdirectory where the
        checkpoints are stored.
    """

    def __init__(
        self,
        *,
        restorees: list[Restoree],
        run_root: str,
        checkpoint_interval: int,
        checkpoint_index: int = 0,
        storage_subdirectory: str = "checkpoints",
    ):
        super().__init__(cycle=checkpoint_interval)
        self._run_root = run_root
        self._checkpoint_index = checkpoint_index
        self._folder_path = os.path.join(run_root, storage_subdirectory)
        self._restorees = restorees

        names = [r.name for r in restorees]
        unique_names = set(names)
        if len(names) != len(unique_names):
            duplicates = {n for n in unique_names if names.count(n) > 1}
            dupe_string = ", ".join(duplicates)
            raise ValueError(
                "Checkpointer is given a list of restorees where\n"
                f"[{dupe_string}]\n"
                "occur multiple times"
            )

    def begin_training(self):
        os.makedirs(self._folder_path, exist_ok=True)

    def end_cycle(self, bp_step, bp_samples):
        name = f"checkpoint_{self._checkpoint_index}.tar"
        final_path = os.path.join(self._folder_path, name)
        state_dict = {
            "callback_state_dicts": {r.name: r.state_dict() for r in self._restorees},
            "training_state": {
                "latest_checkpoint": final_path,
                "bp_step": bp_step,
                "bp_samples": bp_samples,
                "checkpoint_index": self._checkpoint_index,
            },
        }
        torch.save(state_dict, final_path)
        logging.info(f"Saved checkpoint {self._checkpoint_index} at {final_path}.")
        self._checkpoint_index += 1

        return {
            "latest_checkpoint": state_dict["training_state"]["latest_checkpoint"],
            "checkpoint_index": state_dict["training_state"]["checkpoint_index"],
        }


class CheckpointLoader(Callback):
    """CheckpointLoader loads a checkpoint like the one created by
    Checkpointer.

    This is intended for resuming training given a specific checkpoint
    index. It also enables you to load network weights, optimizer, or
    other callback hyper-params independently.  If you want to do
    something more specific, like only restore a specific network
    (outside a callback), it is probably easier to just do it explicitly
    when the network is constructed.

    :param restorees (list[Restoree]): A list of restorees that should
    be restored.
    :param run_root (str): The root path to where the run artifacts
        should be stored.
    :param checkpoint_index (int): Which checkpoint to load.
    :param load_weights (bool): If True, it loads the network weights
    :param load_optimizers (bool): If True, it loads the optimizer state
    :param load_hparams (bool): If True, it loads other callback hyper-
        params
    :param storage_subdirectory (str): The subdirectory where the
        checkpoints are stored.
    """

    def __init__(
        self,
        *,
        restorees: list[Restoree],
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
        self._restorees = restorees

        names = [r.name for r in restorees]
        unique_names = set(names)
        if len(names) != len(unique_names):
            duplicates = {n for n in unique_names if names.count(n) > 1}
            dupe_string = ", ".join(duplicates)
            raise ValueError(
                "Checkpointer is given a list of restorees where\n"
                f"[{dupe_string}]\n"
                "occur multiple times"
            )

    def restore_state(self):
        start_time = time.time()
        if not os.path.exists(self._folder_path):
            raise InvalidCheckpointLocation(
                f"Checkpoint folder {self._folder_path} was specified but does not exist."
            )
        name = f"checkpoint_{self._checkpoint_index}.tar"
        final_path = os.path.join(self._folder_path, name)
        logging.info(f"Loading checkpoints from {self._folder_path}")
        state_dict: dict = torch.load(final_path)

        for restoree in self._restorees:
            state = state_dict["callback_state_dicts"][restoree.name]
            restoree.load_state_dict(
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
