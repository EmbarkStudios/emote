"""
Proxies are bridges between the world the agent acts in and the algorithm training loop.
"""

from __future__ import annotations
from typing import Protocol

from torch import nn

from emote.typing import AgentId, DictObservation, DictResponse


class AgentProxy(Protocol):
    """The interface between the agent in the game and the network used during training."""

    def __call__(
        self,
        obserations: dict[AgentId, DictObservation],
    ) -> dict[AgentId, DictResponse]:
        """Take observations for the active agents and returns the relevant network output."""
        ...

    @property
    def policy(self) -> nn.Module:
        pass

    @property
    def input_names(self) -> tuple[str]:
        ...


class MemoryProxy(Protocol):
    """The interface between the agent in the game and the memory buffer the network trains from."""

    def add(
        self,
        observations: dict[AgentId, DictObservation],
        responses: dict[AgentId, DictResponse],
    ):
        """Store episodes in the memory buffer used for training.

        This is useful e.g. if the data collection is running from a checkpointed model running on
        another machine."""
        ...
