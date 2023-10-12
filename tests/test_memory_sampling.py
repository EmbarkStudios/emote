from collections import defaultdict
import os
import stat
import numpy as np

import pytest
import torch

from emote.memory.coverage_based_strategy import CoverageBasedSampleStrategy
from emote.memory.uniform_strategy import UniformSampleStrategy
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace

from emote.memory.builder import DictObsNStepTable

def create_sample_space() -> MDPSpace:
    reward_space = BoxSpace(dtype=np.float32, shape=(1,))
    action_space = BoxSpace(dtype=np.int32, shape=(1,))
    obs_space = BoxSpace(dtype=np.float32, shape=(2,))
    state_space_dict = {"obs": obs_space}
    state_space = DictSpace(spaces=state_space_dict)
    return MDPSpace(rewards=reward_space, actions=action_space, state=state_space)


def test_memory_export(tmpdir):
    device = torch.device("cpu")
    space = create_sample_space()
    table_max_len = 50_000
    # table_max_len = 8
    table = DictObsNStepTable(spaces=space, 
                              use_terminal_column=False, 
                              maxlen=table_max_len, 
                            #   sampler=UniformSampleStrategy(),
                              sampler=CoverageBasedSampleStrategy(),
                              device=device)

    wave_length = int(table_max_len / 2)

    # Wave 1
    for i in range(wave_length):
        sequence_len = 10
        sequence = {'obs': [np.random.rand(2) for _ in range(sequence_len+1)],
                    'actions': [np.random.rand(1) for _ in range(sequence_len)],
                    'rewards': [np.random.rand(1) for _ in range(sequence_len)]}

        table.add_sequence(identity=i,
                           sequence=sequence,)

    for i in range(1024):
        table.sample(count=1024, sequence_length=5)



    assert False
