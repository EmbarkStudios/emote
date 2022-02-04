"""
Proxies are bridges between the world the agent acts in and the algorithm training loop.
"""

from typing import List, Mapping, Protocol, Tuple
from numpy.typing import ArrayLike


AgentId = str  # Each agent must have its own unique agent id.
ObservationSpace = str  # A name associated with this group of observations.
Observation = Mapping[ObservationSpace, ArrayLike]
Observations = Mapping[AgentId, Observation]
RewardSpace = str
Reward = float
Rewards = Mapping[RewardSpace, Reward]
ResponseSpace = str  # E.g. "actions", "q-value", etc. Not called actions because we also respond with q-values etc.
Response = Mapping[ResponseSpace, ArrayLike]
Responses = Mapping[AgentId, Response]
Episode = List[Tuple[Observation, Response, Reward]]
Episodes = Mapping[AgentId, Episode]


class AgentProxy(Protocol):
    """The interface between the agent in the game and the network used during training."""

    def infer(
        self, dict_obs: Observations, last_rewards: Rewards, save_to_memory: bool = True
    ) -> Responses:
        """Infer takes observations for the active agents and returns the relevant network output."""
        ...


class MemoryProxy(Protocol):
    """The MemoryProxy allows the world to populate the memory buffer that the algorithm trains from."""

    def store(self, episodes: Episodes):
        """Store episodes in the memory buffer used for training.

        This is useful e.g. if the data collection is running from a checkpointed model running on
        another machine."""
        ...
