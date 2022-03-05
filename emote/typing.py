from dataclasses import dataclass
from enum import Enum
from numpy.typing import ArrayLike

# The AgentId is an application-defined integer
AgentId = int
# HiveData is a single ndarray containing correlated data for one agent.
HiveData = ArrayLike
# BatchArray is a concatenated set of arrays from multiple agents.
# The shape of BatchedData will be [Number of Agents, *(shape of HiveData)]
BatchedData = ArrayLike

# Input is a set of named inputs from one agent. We mainly use this for observations.
InputSpace = str
Input = dict[InputSpace, HiveData]
# Input gathers inputs from multiple agents
InputGroup = dict[AgentId, Input]
# InputBatch is the result of merging an InputGroup based on input name.
InputBatch = dict[InputSpace, BatchedData]

# Output is a set of named outputs for one agent
OutputSpace = str
Output = dict[OutputSpace, HiveData]
# Input gathers inputs from multiple agents
OutputGroup = dict[AgentId, Output]
# OutputBatch is the result of evaluating the neural network on an input batch, before unmerging.
OutputBatch = dict[OutputSpace, BatchedData]


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
FloatList = list[float]


class MetaData:
    info: dict[str, float]
    info_lists: dict[str, FloatList]


@dataclass
class HiveObservation:
    rewards: dict[str, float]
    episode_state: EpisodeState
    array_data: dict[str, HiveData]
    metadata: MetaData = None


@dataclass
class HiveResponse:
    list_data: dict[str, FloatList]
    scalar_data: dict[str, float]
