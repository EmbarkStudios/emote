import os

from typing import List, Optional

import torch

from torch import nn, optim

from emote.callback import Callback


class Checkpointer(Callback):
    """Checkpointer writes out a checkpoint every n steps.

    Exactly what is written to the checkpoint is determined by the networks and
    callbacks supplied in the constructor.

    :param callbacks (List[Callback]): A list of callbacks the should be saved.
    :param run_root (str): The root path to where the run artifacts should be stored.
    :param checkpoint_interval (int): Number of backprops between checkpoints.
    :param optimizers (Optional[List[optim.Optimizer]]): Optional list of optimizers
        to save. Usually optimizers are handled by their respective callbacks but
        if you give them to this list they will be handled explicitly.
    :param networks (Optional[List[nn.Module]]): An optional list of networks that
        should be saved. Usually networks and optimizers are both restored by the
        callbacks which handles their parameters.
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
        optimizers: Optional[List[optim.Optimizer]] = None,
        networks: Optional[List[nn.Module]] = None,
        storage_subdirectory: str = "checkpoints",
    ):
        super().__init__(cycle=checkpoint_interval)
        self._cbs = callbacks
        self._run_root = run_root
        self._checkpoint_index = checkpoint_index
        self._opts: List[optim.Optimizer] = optimizers if optimizers else []
        self._nets: List[nn.Module] = networks if networks else []
        self._folder_path = os.path.join(run_root, storage_subdirectory)

    def begin_training(self):
        os.makedirs(self._folder_path, exist_ok=True)

    def end_cycle(self):
        state_dict = {}
        state_dict["callback_state_dicts"] = [cb.state_dict() for cb in self._cbs]
        state_dict["network_state_dicts"] = [net.state_dict() for net in self._nets]
        state_dict["optim_state_dicts"] = [opt.state_dict() for opt in self._opts]
        state_dict["training_state"] = {
            "checkpoint_index": self._checkpoint_index,
        }
        name = f"checkpoint_{self._checkpoint_index}.tar"
        final_path = os.path.join(self._folder_path, name)
        torch.save(state_dict, final_path)
        self._checkpoint_index += 1


class CheckpointLoader(Callback):
    """CheckpointLoader loads a checkpoint like the one created by Checkpointer.

    This is intended for resuming training given a specific checkpoint index. If you
    want to do something more specific, like only restore a specific network, it is
    probably easier to just do it explicitly when the network is constructed.

    :param callbacks (List[Callback]): A list of callbacks the should be restored.
    :param run_root (str): The root path to where the run artifacts should be stored.
    :param checkpoint_index (int): Which checkpoint to load.
    :param reset_training_steps (bool): If True, start training at bp_steps=0 etc.
        Otherwise start the training at whatever step and state the checkpoint has
        saved.
    :param optimizers (Optional[List[optim.Optimizer]]): Optional list of optimizers
        to restore. Usually optimizers are handled by their respective callbacks but
        if you give them to this list they will be handled explicitly.
    :param networks (Optional[List[nn.Module]]): An optional list of networks that
        should be restored. Usually networks and optimizers are both restored by the
        callbacks which handles their parameters.
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
        optimizers: Optional[List[optim.Optimizer]] = None,
        networks: Optional[List[nn.Module]] = None,
        storage_subdirectory: str = "checkpoints",
    ):
        super().__init__()
        self._cbs = callbacks
        self._run_root = run_root
        self._checkpoint_index = checkpoint_index
        self._reset_training_steps = reset_training_steps
        self._opts: List[optim.Optimizer] = optimizers if optimizers else []
        self._nets: List[nn.Module] = networks if networks else []
        self._folder_path = os.path.join(run_root, storage_subdirectory)

    def begin_training(self):
        if not os.path.exists(self._folder_path):
            raise InvalidCheckpointLocation(
                f"Checkpoint folder {self._folder_path} was specified but does not exist."
            )
        name = f"checkpoint_{self._checkpoint_index}.tar"
        final_path = os.path.join(self._folder_path, name)
        state_dict: dict = torch.load(final_path)
        for cb, state in zip(self._cbs, state_dict["callback_state_dicts"]):
            cb.load_state_dict(state)
        for net, state in zip(self._nets, state_dict["network_state_dicts"]):
            net.load_state_dict(state)
        for opt, state in zip(self._opts, state_dict["optim_state_dicts"]):
            opt.load_state_dict(state)
        if self._reset_training_steps:
            return {}
        return state_dict.get("training_state", {})


class InvalidCheckpointLocation(ValueError):
    pass
