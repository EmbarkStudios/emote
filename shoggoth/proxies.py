"""
Proxies are bridges between the world the agent acts in and the algorithm training loop.
"""

from dataclasses import dataclass
from typing import Iterable, OrderedDict, Protocol
from numpy.typing import ArrayLike


AgentId = str  # Each agent must have its own unique agent id.
ObservationSpace = str  # A name associated with this group of observations.
Observations = OrderedDict[ObservationSpace, ArrayLike]
Actions = ArrayLike
Rewards = ArrayLike


@dataclass
class Transitions:
    state: Observations
    actions: Actions
    rewards: Rewards
    agents: Iterable[AgentId]
    dones: Iterable[bool]


class AgentProxy(Protocol):
    """The interface between the agent in the game and the network used during training."""

    def __call__(self, dict_obs: Observations) -> ArrayLike:
        """Infer takes observations for the active agents and returns the relevant network output."""
        ...


class MemoryProxy(Protocol):
    """The MemoryProxy allows the world to populate the memory buffer that the algorithm trains from."""

    def add(self, transition: Transitions):
        """Store episodes in the memory buffer used for training.

        This is useful e.g. if the data collection is running from a checkpointed model running on
        another machine."""
        ...
