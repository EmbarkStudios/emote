"""

"""

import random

from typing import Sequence

import numpy as np

from .core_types import SamplePoint
from .strategy import EjectionStrategy, SampleStrategy, Strategy


class UniformStrategyBase(Strategy):
    """A sampler intended to sample uniformly across the whole set of
    experiences. This base class is used by both the uniform sample and ejection
    strategies."""

    def __init__(self):
        super().__init__()
        self._identities = {}
        self._ids = []
        self._prios = []
        self._dirty = False

    def track(self, identity: int, sequence_length: int):
        self._dirty = True
        self._identities[identity] = sequence_length

    def forget(self, identity: int):
        self._dirty = True
        del self._identities[identity]

    def _rebalance(self):
        self._dirty = False
        self._prios = np.array(tuple(self._identities.values())) / sum(
            self._identities.values()
        )
        self._ids = np.array(tuple(self._identities.keys()), dtype=np.int64)

    def post_import(self):
        original_ids = self._identities.copy()
        for id, length in original_ids.items():
            self.forget(id)
            if id >= 0:
                self.track(-abs(id) - 1, length)

        # rebalance here so we don't have to start by rebalancing all imported
        # memory on the first sample. Not required and the rebalance should be
        # cheap, but this makes the initial state of memory be clean.
        if self._dirty:
            self._rebalance()


################################################################################


class UniformSampleStrategy(UniformStrategyBase, SampleStrategy):
    def sample(self, count: int, transition_count: int) -> Sequence[SamplePoint]:
        if self._dirty:
            self._rebalance()

        identities = np.random.choice(self._ids, size=count, p=self._prios)
        ids = self._identities
        output = []
        app = output.append
        r = random.random
        tm1 = transition_count - 1
        for k in identities:
            offset = int(r() * (ids[k] - tm1))
            app((k, offset, offset + transition_count))

        return output


################################################################################


class UniformEjectionStrategy(UniformStrategyBase, EjectionStrategy):
    def sample(self, count: int) -> Sequence[int]:
        if self._dirty:
            self._rebalance()

        identities = set()
        while count > 0:
            identity = np.random.choice(self._ids, size=1, p=self._prios)[0]

            if identity in identities:
                continue

            length = self._identities[identity]
            count -= length
            identities.add(identity)

        return list(identities)
