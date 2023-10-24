import random
from timeit import timeit
from typing import List
import numpy as np
from emote.memory.builder import DictObsNStepTable
from emote.memory.coverage_based_strategy import CoverageBasedSampleStrategy, CoverageBasedSampleStrategy2
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace

# The episode lengths that are possible
SEQUENCE_RANGE = [10, 1000] 

# The length of the sequences we add to the table
SEQUENCE_LEN = 16  

# The amount of times we sample from the table
SAMPLE_AMOUNT = 8  

# The amount of sequences to sample from the table
COUNT = 64  

# The amount of sequences to add to the table, 
# start is the starting identity, end is the ending identity
START = 0
END = 1_000  

# The maximum size of the table itself
MAXLEN = 500_000

# The modes we want to test
MODES = [0, 1]


def create_sample_space() -> MDPSpace:
    reward_space = BoxSpace(dtype=np.float32, shape=(1,))
    action_space = BoxSpace(dtype=np.int32, shape=(1,))
    obs_space = BoxSpace(dtype=np.float32, shape=(2,))
    state_space_dict = {"obs": obs_space}
    state_space = DictSpace(spaces=state_space_dict)
    return MDPSpace(rewards=reward_space, actions=action_space, state=state_space)


def populate_table(table: DictObsNStepTable):
    sequence_len = random.randint(SEQUENCE_RANGE[0], SEQUENCE_RANGE[1])
    for i in range(START, END):
        sequence = {
            "obs": [np.random.rand(2) for _ in range(sequence_len + 1)],
            "actions": [np.random.rand(1) for _ in range(sequence_len)],
            "rewards": [np.random.rand(1) for _ in range(sequence_len)],
        }

        table.add_sequence(
            identity=i,
            sequence=sequence,
        )


def sample_table(table: DictObsNStepTable):
    for _ in range(SAMPLE_AMOUNT):
        table.sample(COUNT, SEQUENCE_LEN)


def create_table(mode: int): 
    space = create_sample_space()
    sampler = CoverageBasedSampleStrategy()
    if mode == 1:
        sampler = CoverageBasedSampleStrategy2()
    table = DictObsNStepTable(
        spaces=space,
        use_terminal_column=False,
        maxlen=MAXLEN,  
        sampler=sampler,
        device="cpu",
    )
    return table


def test_table_operations():
    for mode in MODES: 
        print(f"Mode: {mode}")
        table = create_table(mode)

        # Benchmarking add_sequence
        time_taken = timeit(lambda: populate_table(table), number=1)
        print(f"Time taken to add: {time_taken} seconds")

        # Benchmarking _rebalance
        time_taken = timeit(lambda: table._sampler._rebalance(), number=1)
        print(f"Time taken to rebalance: {time_taken} seconds")

        # Benchmarking sample
        time_taken = timeit(lambda: sample_table(table), number=1)
        print(f"Time taken to sample: {time_taken} seconds")

    assert False

