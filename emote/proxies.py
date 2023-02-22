"""
Proxies are bridges between the world the agent acts in and the algorithm training loop.
"""

import time

from typing import Dict, Protocol

from emote.callbacks import LoggingCallback
from emote.typing import AgentId, DictObservation, DictResponse


class DictAgentProxy(LoggingCallback):
    """The interface between the agent in the game and the network used during training."""

    def __call__(
        self,
        obserations: Dict[AgentId, DictObservation],
    ) -> Dict[AgentId, DictResponse]:
        """Take observations for the active agents and returns the relevant network output."""
        raise NotImplementedError

    def begin_batch(self):
        return {"completed_inferences": self.completed_inferences}

    def begin_cycle(self, completed_samples):
        super().begin_cycle()
        self._cycle_start_infs = self.completed_inferences
        self._cycle_start_samp = completed_samples
        self._cycle_start_time = time.perf_counter()

    def end_cycle(self, completed_samples):
        super().end_cycle()
        cycle_time = time.perf_counter() - self._cycle_start_time
        cycle_infs = self.completed_inferences - self._cycle_start_infs
        cycle_samp = completed_samples - self._cycle_start_samp
        self.log_scalar("training/inf_per_sec", cycle_infs / cycle_time)
        self.log_scalar("training/samples_per_sec", cycle_samp / cycle_time)
        self.log_scalar("training/samples_per_inf", cycle_samp / cycle_infs)


class MemoryProxy(Protocol):
    """The interface between the agent in the game and the memory buffer the network trains from."""

    def add(
        self,
        observations: Dict[AgentId, DictObservation],
        responses: Dict[AgentId, DictResponse],
    ):
        """Store episodes in the memory buffer used for training.

        This is useful e.g. if the data collection is running from a checkpointed model running on
        another machine."""
        ...
