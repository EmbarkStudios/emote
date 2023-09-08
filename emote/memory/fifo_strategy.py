"""

"""

import random

from collections import deque
from typing import Sequence

from .core_types import SamplePoint
from .strategy import EjectionStrategy, SampleStrategy, Strategy


class FifoStrategyBase(Strategy):
    """A sampler intended to sample in a first-in-first-out style across the whole
    set of experiences. This base class is used by both the fifo sample and
    ejection strategies.
    """

    def __init__(self):
        """Create a FIFO-based strategy."""
        super().__init__()
        self._sequence_lengths = {}
        self._identities = deque()

    def track(self, identity: int, sequence_length: int):
        # N.b. this is most likely a client bug causing us to have a repeat ID,
        # but it can occur when stopping/starting a data generator

        if self._in_simple_import:
            return

        assert identity not in self._sequence_lengths
        self._identities.append(identity)
        self._sequence_lengths[identity] = sequence_length

    def forget(self, identity: int):
        self._identities.remove(identity)
        del self._sequence_lengths[identity]

    def post_import(self):
        original_ids = self._identities.copy()
        for id in original_ids:
            length = self._sequence_lengths[id]
            self.forget(id)
            if id >= 0:
                # This is a guard to prevent recursive memory imports/exports,
                # as that'd make it very hard to uphold variants over time.
                self.track(-abs(id) - 1, length)

    def state(self) -> dict:
        """Serialize the strategy to a JSON-serializable dictionary"""
        return {
            "identities": list(self._identities),
            "sequence_lengths": list(self._sequence_lengths.items()),
        }

    def load_state(self, state: dict):
        """Load the strategy from a dictionary"""
        self._identities = deque(state["identities"])
        self._sequence_lengths = dict(state["sequence_lengths"])


################################################################################


class FifoSampleStrategy(FifoStrategyBase, SampleStrategy):
    def __init__(self, per_episode: bool = True, random_offset: bool = True):
        """Create a FIFO-based sample strategy.

        :param per_episode: if true, will only sample each episode once in a single pass
        :param random_offset: if true will sample at a random offset in each
                              episode. Will be assumed true when sampling per episode
        """
        super().__init__()
        self._per_episode = per_episode
        self._random_offset = random_offset

    def sample(self, count: int, transition_count: int) -> Sequence[SamplePoint]:
        number_episodes = len(self._identities)
        points = []

        if self._per_episode:
            for current_episode_offset in range(count):
                current_episode_id = self._identities[
                    current_episode_offset % number_episodes
                ]
                offset = random.randint(
                    0, self._sequence_lengths[current_episode_id] - transition_count
                )
                points.append((current_episode_id, offset, offset + transition_count))

        else:
            current_episode_offset = 0
            current_offset = 0

            while len(points) < count:
                current_episode_id = self._identities[
                    current_episode_offset % number_episodes
                ]
                if self._random_offset:
                    offset = random.randint(
                        0, self._sequence_lengths[current_episode_id] - transition_count
                    )
                else:
                    offset = current_offset

                points.append((current_episode_id, offset, offset + transition_count))
                current_offset += transition_count

                if (
                    current_offset + transition_count
                    > self._sequence_lengths[current_episode_id]
                ):
                    current_episode_offset += 1
                    current_offset = 0

        return points


################################################################################


class FifoEjectionStrategy(FifoStrategyBase, EjectionStrategy):
    def sample(self, count: int) -> Sequence[int]:
        identities = []

        current_episode_offset = 0
        while count > 0:
            current_episode_id = self._identities[current_episode_offset]
            current_episode_length = self._sequence_lengths[current_episode_id]

            identities.append(current_episode_id)
            count -= current_episode_length

            current_episode_offset += 1

        return identities
