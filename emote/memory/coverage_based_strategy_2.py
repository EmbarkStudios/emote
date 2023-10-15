"""

"""

import random

from typing import Sequence

import numpy as np

from emote.memory.sumtree import SumTree

from .core_types import SamplePoint
from .strategy import SampleStrategy, Strategy


# TODO: Luc: Rename
class CoverageBasedStrategy2(Strategy):
    """A sampler intended to sample based on coverage of experiences,
    favoring less-visited states. This base class can be used for implementing
    various coverage-based sampling strategies."""

    def __init__(self, alpha=0.5):
        super().__init__()
        self._identities = {}
        self._sample_count = {}
        self._alpha = alpha
        self._sum_tree = SumTree(initial_capacity=1e6)  

    def track(self, identity: int, sequence_length: int):
        sample_frequency = self._sample_count.get(identity, 0) + 1
        combined_priority = sequence_length * (1 / sample_frequency ** self._alpha)
        
        self._identities[identity] = sequence_length
        self._sample_count[identity] = self._sample_count.get(identity, 0)
        self._sum_tree.update(identity, combined_priority)

    def forget(self, identity: int):
        self._sum_tree.remove(identity)
        del self._identities[identity]
        del self._sample_count[identity]

# TODO: Luc: Rename
class CoverageBasedSampleStrategy2(CoverageBasedStrategy2, SampleStrategy):
    def __init__(self, alpha=0.5):
        super().__init__(alpha=alpha)

    def sample(self, count: int, transition_count: int) -> Sequence[SamplePoint]:
        output = []
        app = output.append
        r = random.random
        tm1 = transition_count - 1

        for _ in range(count):
            identity, priority = self._sum_tree.sample()
            self._sample_count[identity] += 1
            new_priority = 1 / (self._sample_count[identity] + 1) ** self._alpha
            self._sum_tree.update(identity, new_priority)
            
            offset = int(r() * (self._identities[identity] - tm1))
            app((identity, offset, offset + transition_count))

        return output



# TODO: Luc: Do we need a CoverageBasedEjectionStrategy?