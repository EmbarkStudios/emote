"""
Proxies are bridges between the world the agent acts in and the algorithm training loop.
"""

from typing import Dict, Protocol

from emote.typing import AgentId, DictObservation, DictResponse


class AgentProxy(Protocol):
    """The interface between the agent in the game and the network used during training."""

    def __call__(
        self,
        obserations: Dict[AgentId, DictObservation],
    ) -> Dict[AgentId, DictResponse]:
        """Take observations for the active agents and returns the relevant network output."""
        ...


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
