"""
Proxies are bridges between the world the agent acts in and the algorithm training loop.
"""

from typing import Protocol

from shoggoth.typing import AgentId, HiveObservation, HiveResponse


class AgentProxy(Protocol):
    """The interface between the agent in the game and the network used during training."""

    def __call__(
        self,
        obserations: dict[AgentId, HiveObservation],
    ) -> dict[AgentId, HiveResponse]:
        """Take observations for the active agents and returns the relevant network output."""
        ...


class MemoryProxy(Protocol):
    """The interface between the agent in the game and the memory buffer the network trains from."""

    def add(
        self,
        observations: dict[AgentId, HiveObservation],
        responses: dict[AgentId, HiveResponse],
    ):
        """Store episodes in the memory buffer used for training.

        This is useful e.g. if the data collection is running from a checkpointed model running on
        another machine."""
        ...
