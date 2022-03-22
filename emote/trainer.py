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
    """The Trainer class manages the main training loop in emote.

    It does so by invoking a bunch of callbacks in a number of different places."""

    state: StateDict
    callbacks: List[Callback]
    dataloader: Iterable
    cycle_length: int

    def __init__(
        self,
        callbacks: List[Callback],
        dataloader: Iterable,
    ):
        self.callbacks = sorted(callbacks, key=lambda cb: cb._order)
        self._cyclic_callbacks = [cb for cb in self.callbacks if cb.cycle > 0]
        self.dataloader = dataloader
        self.state = StateDict()

    def train(self, shutdown_signal: Callable = None):
        """The main training loop.

        This method will wait until the memory is full enough to start sampling, and then start
        running cycles of backprops on batches sampled from the memory.

        :param shutdown_signal: A function that returns True if training shut end, False otherwise.
        """
        shutdown_signal = shutdown_signal or (lambda: False)

        self._begin_training()

        try:

            for bp_step, batch in zip(count(1), self.dataloader):
                batch_size = batch['batch_size']
                self.state["bp_step"] = bp_step
                self.state["bp_samples"] = bp_step * batch_size
                self.state.update(batch)

                if shutdown_signal():
                    raise TrainingShutdownException
                self._begin_cycle(bp_step)
                self._begin_batch()
                self._backward()
                self._end_batch()
                self._end_cycle(bp_step)

        except TrainingShutdownException as ex:
            self._end_training(ex)
        except Exception as ex:
            self._end_training(ex)
            raise ex

    def _begin_training(self):
        for cb in self.callbacks:
            if updated_state := cb.begin_training(**self.state):
                self.state.update(updated_state)

    def _begin_cycle(self, bp_step):
        for cb in self._cyclic_callbacks:
            # Start cycles on 1st step of new cycle
            if (bp_step - 1) % cb.cycle == 0:
                if updated_state := cb.begin_cycle(**self.state):
                    self.state.update(updated_state)

    def _begin_batch(self):
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

    def _end_cycle(self, bp_step):
        for cb in self._cyclic_callbacks:
            if bp_step % cb.cycle == 0:
                if updated_state := cb.end_cycle(**self.state):
                    self.state.update(updated_state)

    def _end_training(self, exception: Exception):
        for cb in self.callbacks:
            if updated_state := cb.end_training(exception, **self.state):
                self.state.update(updated_state)
