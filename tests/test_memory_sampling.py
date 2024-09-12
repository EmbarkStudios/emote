"""Test to validate the behavior of `CoverageBasedSampleStrategy`. Tests how
the `alpha` parameter influences the sampling distribution between two waves of
data.

Wave 1 and Wave 2: Two separate sets of data points added to the memory. After each wave, a series of samples are drawn from the memory.

Alpha modulates how much the sampling prioritizes less-visited states. A higher alpha results in a stronger bias towards less-visited states.

Intended Behavior:
    - alpha=0.0: Sampling should be approximately uniform, with no strong bias towards either wave.
    - alpha=1.0: Sampling should strongly prioritize the less-visited states (i.e., states from Wave 2 after it is added).
    - Intermediate alpha values (e.g., alpha=0.5) should result in intermediate behaviors.
"""

import numpy as np
import torch

from emote.memory.builder import DictObsNStepMemoryTable
from emote.memory.coverage_based_strategy import CoverageBasedSampleStrategy
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


TABLE_MAX_LEN = 4096
SAMPLE_AMOUNT = 1024
ALPHAS = [0.0, 0.5, 1.0]
SEQUENCE_LEN = 10


def create_sample_space() -> MDPSpace:
    reward_space = BoxSpace(dtype=np.float32, shape=(1,))
    action_space = BoxSpace(dtype=np.int32, shape=(1,))
    obs_space = BoxSpace(dtype=np.float32, shape=(2,))
    state_space_dict = {"obs": obs_space}
    state_space = DictSpace(spaces=state_space_dict)
    return MDPSpace(rewards=reward_space, actions=action_space, state=state_space)


def populate_table(memory_table: DictObsNStepMemoryTable, sequence_len: int, start: int, end: int):
    for i in range(start, end):
        sequence = {
            "obs": [np.random.rand(2) for _ in range(sequence_len + 1)],
            "actions": [np.random.rand(1) for _ in range(sequence_len)],
            "rewards": [np.random.rand(1) for _ in range(sequence_len)],
        }

        memory_table.add_sequence(
            identity=i,
            sequence=sequence,
        )


def sample_table(
    memory_table: DictObsNStepMemoryTable, sample_amount: int, count: int, sequence_length: int
):
    for _ in range(sample_amount):
        memory_table.sample(count, sequence_length)


def test_memory_export():
    device = torch.device("cpu")
    space = create_sample_space()
    for alpha in ALPHAS:
        memory_table = DictObsNStepMemoryTable(
            spaces=space,
            use_terminal_column=False,
            maxlen=TABLE_MAX_LEN,
            sampler=CoverageBasedSampleStrategy(alpha=alpha),
            device=device,
        )

        wave_length = int(TABLE_MAX_LEN / (2 * SEQUENCE_LEN))

        # Wave 1
        populate_table(
            memory_table=memory_table, sequence_len=SEQUENCE_LEN, start=0, end=wave_length
        )
        sample_table(
            memory_table=memory_table, sample_amount=SAMPLE_AMOUNT, count=5, sequence_length=8
        )
        pre_second_wave_sample_counts = memory_table._sampler._sample_count.copy()

        # Wave 2
        populate_table(
            memory_table=memory_table,
            sequence_len=SEQUENCE_LEN,
            start=wave_length,
            end=wave_length * 2,
        )
        sample_table(
            memory_table=memory_table, sample_amount=SAMPLE_AMOUNT, count=5, sequence_length=8
        )

        second_wave_samples = sum(
            memory_table._sampler._sample_count[id] - pre_second_wave_sample_counts.get(id, 0)
            for id in range(wave_length, wave_length * 2)
        )
        total_new_samples = sum(
            memory_table._sampler._sample_count[id] - pre_second_wave_sample_counts.get(id, 0)
            for id in memory_table._sampler._sample_count.keys()
        )

        proportion_second_wave = second_wave_samples / total_new_samples

        if alpha == 0.0:
            assert proportion_second_wave > 0.4
        elif alpha == 0.5:
            assert proportion_second_wave > 0.6
        elif alpha == 1.0:
            assert proportion_second_wave > 0.8
