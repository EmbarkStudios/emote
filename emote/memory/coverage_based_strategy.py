"""

"""

import random

from typing import Sequence

import numpy as np

from .core_types import SamplePoint
from .strategy import EjectionStrategy, SampleStrategy, Strategy


class CoverageBasedStrategy(Strategy):
    """A sampler intended to sample based on coverage of experiences, 
    favoring less-visited states. This base class can be used for implementing 
    various coverage-based sampling strategies."""

    def __init__(self):
        super().__init__()
        self._identities = {}
        self._sample_count = {}
        self._ids = []
        self._prios = []
        self._dirty = False
        self._alpha = 0.5

    def track(self, identity: int, sequence_length: int):
        self._dirty = True
        self._identities[identity] = sequence_length
        self._sample_count[identity] = self._sample_count.get(identity, 0)

    def forget(self, identity: int):
        self._dirty = True
        del self._identities[identity]
        del self._sample_count[identity]

    def _rebalance(self):
        self._dirty = False
        original_prios = np.array(tuple(self._identities.values())) / sum(self._identities.values())
        self._ids = np.array(tuple(self._identities.keys()), dtype=np.int64)
        
        sample_prios = np.array([1 / (self._sample_count[id] + 1) ** self._alpha for id in self._ids])
        combined_prios = original_prios * sample_prios  
        
        sum_prios = sum(combined_prios)
        self._prios = combined_prios / sum_prios  


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


class CoverageBasedSampleStrategy(CoverageBasedStrategy, SampleStrategy):
    def sample(self, count: int, transition_count: int, alpha=0.5) -> Sequence[SamplePoint]:
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


class CoverageBasedEjectionStrategy(CoverageBasedStrategy, EjectionStrategy):
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
