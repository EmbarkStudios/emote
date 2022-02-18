from typing import Dict, List, Union
import os
import logging
import time

import torch
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter

from .callback import Callback
from .memory import BaseTable
from .utils import BlockTimers


class LoggingCallback(Callback):
    """A Callback that accept logging calls.

    Logged data is saved on this object and gets written
    by the Logger callback."""

    def __init__(self):
        super().__init__()
        self.scalar_logs: Dict[str, Union[float, torch.Tensor]] = {}
        self.hist_logs: Dict[str, Union[float, torch.Tensor]] = {}

    def log_scalar(self, key: str, value: Union[float, torch.Tensor]):
        if isinstance(value, torch.Tensor):
            self.scalar_logs[key] = value.item()
        else:
            self.scalar_logs[key] = value

    def log_histogram(self, key: str, value: torch.Tensor):
        self.hist_logs[key] = value.detach()


class LossCallback(LoggingCallback):
    """Losses are callbacks that implement a *loss function*.

    Loss functions must return a Tensor a batch x scalar dimensions."""

    def __init__(
        self,
        name: str,
        optimizer: optim.Optimizer,
        max_grad_norm: float,
        data_group: str,
    ):
        super().__init__()
        self.data_group = data_group
        self.name = name
        self.optimizer = optimizer
        self.parameters = [
            p for param_group in optimizer.param_groups for p in param_group["params"]
        ]
        self._max_grad_norm = max_grad_norm

    # def setup_schedules(self, total_timesteps): #TODO(singhblom) implement correct schedule for torch
    #     self._lr_schedule.setup(total_timesteps)

    def backward(self, *args, **kwargs):
        self.optimizer.zero_grad()
        loss = self.loss(*args, **kwargs)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters, self._max_grad_norm)
        self.optimizer.step()

        self.log_scalar(f"loss/{self.name}_loss", loss)
        self.log_scalar(f"loss/{self.name}_gradient_norm", grad_norm)

    def end_batch(self):
        self.log_scalar(
            f"training/{self.name}_learning_rate", self.optimizer.param_groups[0]["lr"]
        )

    def get_state(self):
        return {"optimizer_state_dict": self.optimizer.state_dict()}

    def load_state(self, state):
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

    @Callback.extend
    def loss(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class TensorboardLogger(Callback):
    def __init__(
        self,
        callbacks: List[LoggingCallback],
        writer: SummaryWriter,
        bp_step: bool = True,
        bp_samples: bool = True,
    ):
        super().__init__()
        self._callbacks = callbacks
        self._writer = writer
        self._step = bp_step
        self._samples = bp_samples

    def log_scalars(self, step, suffix=None):
        """Logs scalar logs adding optional suffix on the first level.
        E.g. If k='training/loss' and suffix='bp_step', k will be renamed to 'training_bp_step/loss'."""
        for cb in self._callbacks:
            for k, v in cb.scalar_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                self._writer.add_scalar(k, v, step)

    def end_batch(self, should_bp_log, bp_step, bp_samples):
        if should_bp_log and self._step:
            self.log_scalars(bp_step, suffix="bp_step")
        if should_bp_log and self._samples:
            self.log_scalars(bp_samples, suffix="bp_samples")


class TerminalLogger(Callback):
    def __init__(
        self,
        callbacks: List[LoggingCallback],
        bp_log_interval: int,
    ):
        super().__init__()
        self._callbacks = callbacks
        self._bp_log_interval = bp_log_interval

    def log_scalars(self, step, suffix=None):
        """Logs scalar logs adding optional suffix on the first level.
        E.g. If k='training/loss' and suffix='bp_step', k will be renamed to 'training_bp_step/loss'."""
        for cb in self._callbacks:
            for k, v in cb.scalar_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                logging.info("%s@%s:\t%.4f", k, step, v)

    def end_batch(self, bp_step):
        if bp_step % self._bp_log_interval == 0:
            self.log_scalars(bp_step)


class Checkpointer(Callback):
    def __init__(
        self,
        net: nn.Module,
        callbacks: List[LoggingCallback],
        path: str,
    ):
        super().__init__()
        self._net = net
        self._callbacks = callbacks
        self._path = path

    def end_cycle(self, cycle_index, inf_step, bp_step, bp_samples):
        state_dict = {}
        state_dict["model_state_dict"] = self._net.state_dict()
        state_dict["callbacks"] = [cb.get_state() for cb in self._callbacks]
        state_dict["cycle_index"] = cycle_index
        state_dict["inf_step"] = inf_step
        state_dict["bp_step"] = bp_step
        state_dict["bp_samples"] = bp_samples
        torch.save(state_dict, f"{self._path}.{cycle_index}.tar")


class CheckpointLoader(Callback):
    def __init__(
        self,
        net: nn.Module,
        callbacks: List[LoggingCallback],
        path: str,
        cycle_index: int,
        reset_training_steps: bool = False,
    ):
        super().__init__()
        self._net = net
        self._callbacks = callbacks
        self._path = path
        self._cycle_index = cycle_index
        self._reset_training_steps = reset_training_steps

    def begin_training(self):
        state_dict = torch.load(f"{self._path}.{self._cycle_index}.tar")
        self._net.load_state_dict(state_dict["model_state_dict"])
        for cb, state in zip(self._callbacks, state_dict["callbacks"]):
            cb.load_state(state)
        if self._reset_training_steps:
            return {}
        return {
            "cycle_index": state_dict["cycle_index"],
            "inf_step": state_dict["inf_step"],
            "bp_step": state_dict["bp_step"],
            "bp_samples": state_dict["bp_samples"],
        }


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
    def begin_training(self, memory: BaseTable):
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
        memory: BaseTable = None,
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
