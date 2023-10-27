import numpy as np
import pytest

from emote.memory.column import Column
from emote.memory.fifo_strategy import FifoEjectionStrategy
from emote.memory.memory import JointMemoryLoader, JointMemoryLoaderWithDataGroup, MemoryLoader
from emote.memory.table import ArrayTable
from emote.memory.uniform_strategy import UniformSampleStrategy


@pytest.fixture
def a_dummy_table():
    tab = ArrayTable(
        columns=[Column("state", (), np.float32), Column("action", (), np.float32)],
        maxlen=1_000,
        sampler=UniformSampleStrategy(),
        ejector=FifoEjectionStrategy(),
        length_key="action",
        device="cpu",
    )
    tab.add_sequence(
        0,
        {
            "state": [5.0, 6.0],
            "action": [1.0],
        },
    )

    return tab


@pytest.fixture
def another_dummy_table():
    tab = ArrayTable(
        columns=[Column("state", (), np.float32), Column("action", (), np.float32)],
        maxlen=1_000,
        sampler=UniformSampleStrategy(),
        ejector=FifoEjectionStrategy(),
        length_key="action",
        device="cpu",
    )
    tab.add_sequence(
        0,
        {
            "state": [5.0, 6.0],
            "action": [1.0],
        },
    )

    return tab


def test_joint_memory_loader(
    a_dummy_table: ArrayTable, another_dummy_table: ArrayTable
):
    a_loader = MemoryLoader(
        table=a_dummy_table,
        rollout_count=1,
        rollout_length=1,
        size_key="batch_size",
        data_group="a",
    )
    another_loader = MemoryLoader(
        table=another_dummy_table,
        rollout_count=1,
        rollout_length=1,
        size_key="batch_size",
        data_group="another",
    )

    joint_loader = JointMemoryLoader(loaders=[a_loader, another_loader])

    data = next(iter(joint_loader))
    assert "a" in data and "another" in data, "JointMemoryLoader did not yield expected memory data"


def test_joint_memory_loader_datagroup(a_dummy_table: ArrayTable, another_dummy_table: ArrayTable):
    a_loader = MemoryLoader(
        table=a_dummy_table,
        rollout_count=1,
        rollout_length=1,
        size_key="batch_size",
        data_group="a",
    )
    another_loader = MemoryLoader(
        table=another_dummy_table,
        rollout_count=1,
        rollout_length=1,
        size_key="batch_size",
        data_group="another",
    )

    joint_loader = JointMemoryLoaderWithDataGroup(
        loaders=[a_loader, another_loader], data_group="joint_datagroup"
    )

    encapsulated_data = next(iter(joint_loader))
    data = encapsulated_data["joint_datagroup"]

    assert (
        "joint_datagroup" in encapsulated_data
    ), "Expected joint dataloader to place data in its own datagroup, but it does not exist."
    assert (
        "a" in data and "another" in data
    ), "Expected joint dataloader to actually place data in its datagroup, but it is empty."
