import time
from typing import Any, Callable, Iterable, List, MutableMapping
from weakref import ref
from itertools import count

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

    def train(self, shutdown_signal: Callable):
        """The main training loop.

        This method will wait until the memory is full enough to start sampling, and then start
        running cycles of backprops on batches sampled from the memory.

        :param shutdown_signal: A function that returns True if training shut end, False otherwise.
        """
        shutdown_signal = shutdown_signal or (lambda: False)

        while not self.dataloader.is_ready():
            time.sleep(0.1)

        bp_step = 0

        self.begin_training()

        try:
            for cycle_index in count():
                self.state["cycle_index"] = cycle_index
                self.begin_cycle()

                for cycle_step, batch in zip(range(self.cycle_length), self.dataloader):
                    self.state["cycle_step"] = cycle_step
                    self.state["bp_step"] = bp_step

                    if shutdown_signal():
                        raise TrainingShutdownException

                    self.begin_batch(batch)
                    self.begin_backward()
                    self.backward()
                    self.end_backward()
                    self.end_batch()

                    bp_step += 1
                self.end_cycle()
        except TrainingShutdownException as ex:
            self.end_training(ex)
        except Exception as ex:
            self.end_training(ex)
            raise ex

    def begin_training(self):
        for cb in self.callbacks:
            if updated_state := cb.begin_training(**self.state):
                self.state.update(updated_state)

    def begin_cycle(self):
        for cb in self.callbacks:
            if updated_state := cb.begin_cycle(**self.state):
                self.state.update(updated_state)

    def begin_batch(self, batch):
        self.state.update(batch)
        for cb in self.callbacks:
            if updated_state := cb.begin_batch(**self.state):
                self.state.update(updated_state)

    def begin_backward(self):
        for cb in self.callbacks:
            if updated_state := cb.begin_backward(**self.state):
                self.state.update(updated_state)

    def backward(self):
        for cb in self.callbacks:
            if updated_state := cb.backward(**self.state):
                self.state.update(updated_state)

    def end_backward(self):
        for cb in self.callbacks:
            if updated_state := cb.end_backward(**self.state):
                self.state.update(updated_state)

    def end_batch(self):
        for cb in self.callbacks:
            if updated_state := cb.end_batch(**self.state):
                self.state.update(updated_state)

    def end_cycle(self):
        for cb in self.callbacks:
            if updated_state := cb.end_cycle(**self.state):
                self.state.update(updated_state)

    def end_training(self, exception: Exception):
        for cb in self.callbacks:
            if updated_state := cb.end_training(exception, **self.state):
                self.state.update(updated_state)
