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

from emote.callback import Callback

from ..typing import AgentId, DictObservation, DictResponse, EpisodeState
from ..utils import TimedBlock
from .core_types import Matrix
from .table import Table


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


class TableMemoryProxy:
    """The sequence builder wraps a sequence-based memory to build full episodes
    from [identity, observation] data. Not thread safe.
    """

    def __init__(
        self,
        table: Table,
        minimum_length_threshold: Optional[int] = None,
        use_terminal: bool = False,
    ):
        self._store: Dict[AgentId, Episode] = {}
        self._table = table
        if minimum_length_threshold is None:
            self._min_length_filter = lambda _: True
        else:
            key = table._length_key
            self._min_length_filter = (
                lambda ep: len(ep[key]) >= minimum_length_threshold
            )

        self._completed_episodes: set[AgentId] = set()
        self._term_states = [EpisodeState.TERMINAL, EpisodeState.INTERRUPTED]
        self._use_terminal = use_terminal

    def size(self):
        return self._table.size()

    def resize(self, new_size):
        self._table.resize(new_size)

    def is_initial(self, identity: int):
        """Returns true if identity is not already used in a partial sequence. Does not
        validate if the identity is associated with a complete episode."""
        return identity not in self._store

    def add(
        self,
        observations: Dict[AgentId, DictObservation],
        responses: Dict[AgentId, DictResponse],
    ):
        completed_episodes = {}

        for agent_id, observation in observations.items():
            data = {space: feature for space, feature in observation.array_data.items()}

            if observation.episode_state != EpisodeState.INITIAL:
                data["rewards"] = observation.rewards["reward"]

            if observation.episode_state in self._term_states:
                if self._use_terminal:
                    # The terminal value assigned here is the terminal _mask_ value,
                    # not whether it is terminal. In this case, our legacy code
                    # treated all terminals as fatal, i.e., truncated bootstrap.
                    # Since this is the terminal mask value, an interrupted
                    # episode should be 1.0 or "infinite bootstrap horizon"
                    data["terminal"] = float(
                        observation.episode_state == EpisodeState.INTERRUPTED
                    )

                if agent_id not in self._store:
                    # First warn that this is a new agent id:
                    if agent_id in self._completed_episodes:
                        logging.warning(
                            "agent_id has already been completed: %d", agent_id
                        )
                    else:
                        logging.warning(
                            "agent_id completed with no previous sequence: %d", agent_id
                        )

                self._completed_episodes.add(agent_id)

                if agent_id not in self._store:
                    # Then continue without sending an empty episode to the table.
                    continue

                ep = self._store.pop(agent_id).complete(data)
                if self._min_length_filter(ep):  # else discard
                    completed_episodes[agent_id] = ep

            else:
                assert (
                    agent_id in responses
                ), "Mismatch between observations and responses!"
                response = responses[agent_id]
                data.update(response.list_data)
                data.update(response.scalar_data)

                if agent_id not in self._store:
                    self._store[agent_id] = Episode.from_initial(data)

                else:
                    self._store[agent_id].append(data)

        for agent_id, sequence in completed_episodes.items():
            self._table.add_sequence(agent_id, sequence)


class MemoryLoader:
    def __init__(
        self,
        table: Table,
        rollout_count: int,
        rollout_length: int,
        size_key: str,
        data_group: str = "default",
    ):
        self.data_group = data_group
        self.table = table
        self.rollout_count = rollout_count
        self.rollout_length = rollout_length
        self.size_key = size_key
        self.timer = TimedBlock()

    def is_ready(self):
        """True if the data loader has enough data to start providing data"""
        return self.table.size() >= (self.rollout_count * self.rollout_length)

    def __iter__(self):
        if not self.is_ready():
            raise Exception(
                "Data loader does not have enough data.\
                 Check `is_ready()` before trying to iterate over data."
            )

        while True:
            with self.timer:
                data = self.table.sample(self.rollout_count, self.rollout_length)

            data[self.size_key] = self.rollout_count * self.rollout_length
            yield {self.data_group: data, self.size_key: data[self.size_key]}


class MemoryWarmup(Callback):
    """A blocker to ensure memory has data.

    This ensures the memory has enough data when training starts, as the memory
    will panic otherwise. This is useful if you use an async data generator.

    If you do not use an async data generator this can deadlock your training
    loop and prevent progress.
    """

    def __init__(self, loader: MemoryLoader):
        super().__init__()
        self._order = 100
        self._loader = loader

    def begin_training(self):
        import time

        while not self._loader.is_ready():
            time.sleep(0.1)
