import time
from typing import Any, Callable, Iterable, List, MutableMapping
from weakref import ref
from itertools import count

from torch.optim import Optimizer

from .callback import Callback
from .utils import WeakReference


class StateDict(
    dict, MutableMapping[str, Any]
):  # TODO(singhblom) Split state dict into two - one persistable and one transient.
    """Wrapped around a dict allowing usage in a weakref"""

    def get_handle(self) -> WeakReference["StateDict"]:
        """Retrieve a weak handle to this state dict, with no promise of ownership
        or lifetime
        """
        return ref(self)


class TrainingShutdownException(Exception):
    pass


class Trainer:
    """The Trainer class manages the main training loop in shoggoth.

    It does so by invoking a bunch of callbacks in a number of different places."""

    state: StateDict
    callbacks: List[Callback]
    dataloader: Iterable
    cycle_length: int

    def __init__(
        self,
        callbacks: List[Callback],
        dataloader: Iterable,
        cycle_length: int,
    ):
        self.callbacks = sorted(callbacks, key=lambda cb: cb._order)
        self.dataloader = dataloader
        self.cycle_length = cycle_length
        self.state = StateDict()

    def train(self, shutdown_signal: Callable = None):
        """The main training loop.

        This method will wait until the memory is full enough to start sampling, and then start
        running cycles of backprops on batches sampled from the memory.

        :param shutdown_signal: A function that returns True if training shut end, False otherwise.
        """
        shutdown_signal = shutdown_signal or (lambda: False)

        bp_step = 0

        self._begin_training()

        try:
            for cycle_index in count():
                self.state["cycle_index"] = cycle_index
                self._begin_cycle()

                for cycle_step, batch in zip(range(self.cycle_length), self.dataloader):
                    self.state["cycle_step"] = cycle_step
                    self.state["bp_step"] = bp_step

                    if shutdown_signal():
                        raise TrainingShutdownException

                    self._begin_batch(batch)
                    self._backward()
                    self._end_batch()

                    bp_step += 1
                self._end_cycle()
        except TrainingShutdownException as ex:
            self._end_training(ex)
        except Exception as ex:
            self._end_training(ex)
            raise ex

    def _begin_training(self):
        for cb in self.callbacks:
            if updated_state := cb.begin_training(**self.state):
                self.state.update(updated_state)

    def _begin_cycle(self):
        for cb in self.callbacks:
            if updated_state := cb.begin_cycle(**self.state):
                self.state.update(updated_state)

    def _begin_batch(self, batch):
        self.state.update(batch)
        for cb in self.callbacks:
            if updated_state := cb.begin_batch(**self.state):
                self.state.update(updated_state)

    def _backward(self):
        for cb in self.callbacks:
            if updated_state := cb.backward(**self.state):
                self.state.update(updated_state)

    def _end_batch(self):
        for cb in self.callbacks:
            if updated_state := cb.end_batch(**self.state):
                self.state.update(updated_state)

    def _end_cycle(self):
        for cb in self.callbacks:
            if updated_state := cb.end_cycle(**self.state):
                self.state.update(updated_state)

    def _end_training(self, exception: Exception):
        for cb in self.callbacks:
            if updated_state := cb.end_training(exception, **self.state):
                self.state.update(updated_state)
