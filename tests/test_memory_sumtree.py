"""
Compare the sumtree implementation with the naive implementation.
This will test the speedup of the sumtree implementation.
"""

import numpy as np
from emote.memory.coverage_based_strategy_2 import CoverageBasedSampleStrategy2
import torch
import time

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace

TABLE_MAX_LEN = 4_000


def create_sample_space() -> MDPSpace:
    reward_space = BoxSpace(dtype=np.float32, shape=(1,))
    action_space = BoxSpace(dtype=np.int32, shape=(1,))
    obs_space = BoxSpace(dtype=np.float32, shape=(2,))
    state_space_dict = {"obs": obs_space}
    state_space = DictSpace(spaces=state_space_dict)
    return MDPSpace(rewards=reward_space, actions=action_space, state=state_space)


# TODO: Luc: Move this to helper function and also in test_memory_sampling.py
def sample_table(table: DictObsNStepTable, sample_amount: int, count: int, sequence_length: int):
    for _ in range(sample_amount):
        table.sample(count, sequence_length)

# TODO: Luc: Move this to helper function and also in test_memory_sampling.py
def populate_table(table: DictObsNStepTable, sequence_len: int, start: int, end: int):
    for i in range(start, end):
        sequence = {
            "obs": [np.random.rand(2) for _ in range(sequence_len + 1)],
            "actions": [np.random.rand(1) for _ in range(sequence_len)],
            "rewards": [np.random.rand(1) for _ in range(sequence_len)],
        }

        table.add_sequence(
            identity=i,
            sequence=sequence,
        )

def test_sumtree_speedup():
    device = torch.device("cpu")
    space = create_sample_space()
    table = DictObsNStepTable(
        spaces=space,
        use_terminal_column=False,
        maxlen=TABLE_MAX_LEN,
        sampler=CoverageBasedSampleStrategy2(),
        device=device,
    )

    start_time = time.time()

    populate_table(table=table, sequence_len=200, start=0, end=TABLE_MAX_LEN)
    print(table.size())
    print(f"Populate table took {time.time() - start_time} seconds")

    start_time = time.time()
    sample_table(table=table, sample_amount=2000, count=5, sequence_length=8)
    print(f"Sample table took {time.time() - start_time} seconds")
    assert False
    # sample_table(table=table, sample_amount=SAMPLE_AMOUNT, count=5, sequence_length=8)

