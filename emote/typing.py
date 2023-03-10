from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Union

import numpy as np
import torch

from numpy.typing import ArrayLike


TensorType = Union[torch.Tensor, np.ndarray]

RewardFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TermFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# The AgentId is an application-defined integer
AgentId = int
# SingleAgentData is a single ndarray containing correlated data for one agent.
SingleAgentData = ArrayLike
# BatchedData is a concatenated set of arrays from multiple agents.
# The shape of BatchedData will be [Number of Agents, *(shape of SingleAgentData)]
BatchedData = ArrayLike

# Input is a set of named inputs from one agent. We mainly use this for observations.
InputSpace = str
Input = Dict[InputSpace, SingleAgentData]
# Input gathers inputs from multiple agents
InputGroup = Dict[AgentId, Input]
# InputBatch is the result of merging an InputGroup based on input name.
InputBatch = Dict[InputSpace, BatchedData]

# Output is a set of named outputs for one agent
OutputSpace = str
Output = Dict[OutputSpace, SingleAgentData]
# Input gathers inputs from multiple agents
OutputGroup = Dict[AgentId, Output]
# OutputBatch is the result of evaluating the neural network on an input batch, before unmerging.
OutputBatch = Dict[OutputSpace, BatchedData]


class EpisodeState(Enum):
    # The agent expects an action back and can continue to at least one more state
    RUNNING = 0

    # The episode has ended due to external factors the agent has no ability to
    # affect - for example, the agent timing out or the game round ending.
    INTERRUPTED = 1

    # The episode has ended due to events the agent could have affected, and
    # should learn to understand.
    TERMINAL = 2

    # This is the first step of an agent's lifetime. Sending this multiple
    # times for one agent is an error.
    INITIAL = 3


# In the future we might switch to supporting flat np.arrays here.
FloatList = List[float]


@dataclass
class MetaData:
    info: Dict[str, float]
    info_lists: Dict[str, FloatList]


@dataclass
class DictObservation:
    rewards: Dict[str, float]
    episode_state: EpisodeState
    array_data: Dict[str, SingleAgentData]
    metadata: MetaData = None


@dataclass
class DictResponse:
    list_data: Dict[str, FloatList]
    scalar_data: Dict[str, float]
