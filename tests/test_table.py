import numpy as np
import pytest
import torch

from emote.memory.adaptors import TerminalAdaptor
from emote.memory.fifo_strategy import FifoEjectionStrategy
from emote.memory.storage import SyntheticDones
from emote.memory.table import ArrayTable, Column, TagColumn, VirtualColumn
from emote.memory.uniform_strategy import UniformSampleStrategy


@pytest.fixture
def table():
    spec = [
        Column(name="obs", dtype=np.float32, shape=(3,)),
        Column(name="reward", dtype=np.float32, shape=()),
        VirtualColumn("dones", dtype=bool, shape=(1,), target_name="reward", mapper=SyntheticDones),
        VirtualColumn(
            "masks",
            dtype=np.float32,
            shape=(1,),
            target_name="reward",
            mapper=SyntheticDones.as_mask,
        ),
        TagColumn(name="terminal", shape=(), dtype=np.float32),
    ]

    table = ArrayTable(
        columns=spec,
        maxlen=10_000,
        sampler=UniformSampleStrategy(),
        ejector=FifoEjectionStrategy(),
        length_key="reward",
        adaptors=[TerminalAdaptor("terminal", "masks")],
        device="cpu",
    )

    return table


def test_sampled_data_is_always_copied(table: ArrayTable):
    for ii in range(0, 600):
        table.add_sequence(
            ii,
            dict(
                obs=[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                reward=[1, 2, 3, 4],
                terminal=[0, 0, 0, 0, 0],
            ),
        )

    sample_count = 100
    counts = [256, 512]
    seq_len = 3
    for _ in range(sample_count):
        for count in counts:
            sample1 = table.sample(count, seq_len)
            sample2 = table.sample(count, seq_len)

            for key in sample1.keys():
                col_samp_1: torch.Tensor = sample1[key]
                col_samp_2: torch.Tensor = sample2[key]

                assert (
                    col_samp_1.data_ptr() != col_samp_2.data_ptr()
                ), "2 table samples share memory! This is not allowed! Samples must always copy their data."
