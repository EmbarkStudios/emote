"""

"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence

from .core_types import Matrix, SamplePoint


class Strategy(ABC):
    """A generalized strategy that may be specialized for sampling or ejection from
    a memory buffer"""

    @abstractmethod
    def track(self, identity: int, sequence_length: int):
        """Track a sequence given by identity and sequence_length that exists in the
        memory

        :param identity: an identity that is globally unique
        :param sequence_length: the number of transitions in the sequence
                                identified by identity"""
        ...

    @abstractmethod
    def forget(self, identity: int):
        """Forget the sequence of transitions given by identity"""
        ...

    def on_sample(
        self,
        ids_and_offsets: Sequence[SamplePoint],
        transition_count: int,
        advantages: Matrix | None = None,
    ):
        """Called after a sampling strategy has been invoked, to give the strategy a
        chance to update sampling weights in case it uses prioritized sampling
        """
        ...

    def post_import(self):
        """Post-import validation of invariants and cleanup.

        This *has* to forget any imported negative ids, anything else is
        implementation-defined.
        """
        ...


################################################################################


class SampleStrategy(Strategy):
    """A strategy specialized for sampling"""

    @abstractmethod
    def sample(self, count: int, transition_count: int) -> Sequence[SamplePoint]:
        """Apply the sampling strategy to the memory metadata, returning `count`
        identities and offsets to use when sampling from the memory"""
        ...


################################################################################


class EjectionStrategy(Strategy):
    """A strategy specialized for ejection sampling"""

    @abstractmethod
    def sample(self, count: int) -> Sequence[int]:
        """Apply the sampling strategy to the memory metadata, returning a list of
        identities that shall be ejected from the memory to remove at least "count" transitions.

        """
        ...
