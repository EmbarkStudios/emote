import time
import numpy as np
from emote.memory.builder import DictObsNStepTable
from emote.memory.coverage_based_strategy import CoverageBasedSampleStrategy
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace

# TODO: Luc: Make utils with test_memory_sampling.py
# ---
def create_sample_space() -> MDPSpace:
    reward_space = BoxSpace(dtype=np.float32, shape=(1,))
    action_space = BoxSpace(dtype=np.int32, shape=(1,))
    obs_space = BoxSpace(dtype=np.float32, shape=(2,))
    state_space_dict = {"obs": obs_space}
    state_space = DictSpace(spaces=state_space_dict)
    return MDPSpace(rewards=reward_space, actions=action_space, state=state_space)

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
def sample_table(table: DictObsNStepTable, sample_amount: int, count: int, sequence_length: int):
    for _ in range(sample_amount):
        table.sample(count, sequence_length)
# ---

def test_table_operations():
    sequence_len = 10
    sample_amount = 50
    count = 10
    alpha = 0.5
    start = 0
    end = 100
    device = "cpu"  
    
    space = create_sample_space()
    table = DictObsNStepTable(
        spaces=space,
        use_terminal_column=False,
        maxlen=500_000,  
        sampler=CoverageBasedSampleStrategy(alpha=alpha),
        device=device,
    )

    start_time = time.time()
    populate_table(table, sequence_len, start, end)
    add_time = time.time() - start_time
    print(f"Time taken to add: {add_time} seconds")

    # Benchmarking _rebalance
    start_time = time.time()
    table._sampler._rebalance()  # Assuming the sampler has been set and the _rebalance method is accessible
    rebalance_time = time.time() - start_time
    print(f"Time taken to rebalance: {rebalance_time} seconds")

    # Benchmarking sample
    start_time = time.time()
    sample_table(table, sample_amount, count, sequence_len)
    sample_time = time.time() - start_time
    print(f"Time taken to sample: {sample_time} seconds")
    assert False

