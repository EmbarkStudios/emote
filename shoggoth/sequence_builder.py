"""Sequence builder collates observations into sequences stored in the memory.

The sequence builder is the API between "instant" based APIs such as the agent
proxy and the episode-based functionality of the memory implementation. The goal
of the sequence builder is to consume individual timesteps per agent and collate
them into episodes before submission into the memory.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from embarkrl.memory import Table
from embarkrl.memory.core_types import Matrix

Observations = Mapping[int, Mapping[str, Matrix]]
Terminals = Mapping[int, Mapping[str, Matrix]]


@dataclass
class Episode:
    """An episode of data being constructed"""

    data: Dict[str, List[Matrix]] = field(default_factory=lambda: defaultdict(list))

    def append(self, observation: Mapping[str, Matrix]) -> Tuple:
        for k, v in observation.items():
            self.data[k].append(v)

    def complete(self, observation: Mapping[str, Matrix]) -> Mapping[str, Matrix]:
        self.append(observation)
        return self.data

    @staticmethod
    def from_initial(observation: Mapping[str, Matrix]) -> "Episode":
        episode = Episode()
        episode.append(observation)
        return episode


################################################################################


class SequenceBuilder:
    """The sequence builder wraps a sequence-based memory to build full episodes
    from [identity, observation] data. Not thread safe.
    """

    def __init__(
        self, memory: Table, *, minimum_length_threshold: Optional[int] = None
    ):
        self._store: Dict[int, Episode] = dict()
        self._memory = memory
        if minimum_length_threshold is None:
            self._min_length_filter = lambda _: True
        else:
            key = memory._length_key
            self._min_length_filter = (
                lambda ep: len(ep[key]) >= minimum_length_threshold
            )

        self._completed_episodes = set()

    def is_initial(self, identity):
        """Returns true if identity is not already used in a partial sequence. Does not
        validate if the identity is associated with a complete episode."""
        return identity not in self._store

    def add(self, observations: Observations, terminals: Terminals):
        """Adds a set of data to episode builder.

        :param observations: Dictionary from agent_id to the data associated with timepoint one
        :param results: Dictionary from agent_id to the data associated with timepoint two, completing transitions"""
        completed_episodes = {}

        for identity, observation in observations.items():
            if identity not in self._store:
                self._store[identity] = Episode.from_initial(observation)

            else:
                self._store[identity].append(observation)

        for identity, observation in terminals.items():
            if identity not in self._store:
                if identity in self._completed_episodes:
                    logging.warning("identity has already been completed: %d", identity)
                else:
                    logging.warning(
                        "identity completed with no previous sequence: %d", identity
                    )

            self._completed_episodes.add(identity)

            if identity not in self._store:
                continue

            ep = self._store.pop(identity).complete(observation)
            if self._min_length_filter(ep):  # else discard
                completed_episodes[identity] = ep

        for identity, sequence in completed_episodes.items():
            self._memory.add_sequence(identity, sequence)
