"""
Proxies are bridges between the world the agent acts in and the algorithm training loop.
"""

from dataclasses import dataclass
from typing import OrderedDict, Protocol
from numpy.typing import ArrayLike


AgentId = str  # Each agent must have its own unique agent id.
ObservationSpace = str  # A name associated with this group of observations.
AgentObservation = OrderedDict[AgentId, ArrayLike]
Observations = OrderedDict[ObservationSpace, AgentObservation]
RewardSpace = str
AgentReward = OrderedDict[AgentId, float]
Rewards = OrderedDict[RewardSpace, AgentReward]
ResponseSpace = str  # E.g. "actions", "q-value", etc. Not called actions because we also respond with q-values etc.
AgentResponse = OrderedDict[AgentId, ArrayLike]
Responses = OrderedDict[ResponseSpace, AgentResponse]


@dataclass
class Transitions:
    state: Observations
    actions: ArrayLike
    rewards: Rewards


class AgentProxy(Protocol):
    """The interface between the agent in the game and the network used during training."""

    def __call__(self, dict_obs: Observations) -> Responses:
        """Infer takes observations for the active agents and returns the relevant network output."""
        ...


class TransitionMemoryProxy(Protocol):
    """The MemoryProxy allows the world to populate the memory buffer that the algorithm trains from."""

    def push(self, transition: Transitions):
        """Store episodes in the memory buffer used for training.

        This is useful e.g. if the data collection is running from a checkpointed model running on
        another machine."""
        ...
