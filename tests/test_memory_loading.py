import numpy as np
import pytest

from emote.memory.column import Column
from emote.memory.fifo_strategy import FifoEjectionStrategy
from emote.memory.memory import JointMemoryLoader, JointMemoryLoaderWithDataGroup, MemoryLoader
from emote.memory.table import ArrayMemoryTable
from emote.memory.uniform_strategy import UniformSampleStrategy


@pytest.fixture
def a_dummy_table():
    memory_table = ArrayMemoryTable(
        columns=[Column("state", (), np.float32), Column("action", (), np.float32)],
        maxlen=1_000,
        sampler=UniformSampleStrategy(),
        ejector=FifoEjectionStrategy(),
        length_key="action",
        device="cpu",
    )
    memory_table.add_sequence(
        0,
        {
            "state": [5.0, 6.0],
            "action": [1.0],
        },
    )

    return memory_table


@pytest.fixture
def another_dummy_table():
    memory_table = ArrayMemoryTable(
        columns=[Column("state", (), np.float32), Column("action", (), np.float32)],
        maxlen=1_000,
        sampler=UniformSampleStrategy(),
        ejector=FifoEjectionStrategy(),
        length_key="action",
        device="cpu",
    )
    memory_table.add_sequence(
        0,
        {
            "state": [5.0, 6.0],
            "action": [1.0],
        },
    )

    return memory_table


def test_joint_memory_loader(
    a_dummy_table: ArrayMemoryTable, another_dummy_table: ArrayMemoryTable
):
    a_loader = MemoryLoader(
        memory_table=a_dummy_table,
        rollout_count=1,
        rollout_length=1,
        size_key="batch_size",
        data_group="a",
    )
    another_loader = MemoryLoader(
        memory_table=another_dummy_table,
        rollout_count=1,
        rollout_length=1,
        size_key="batch_size",
        data_group="another",
    )

    joint_loader = JointMemoryLoader(loaders=[a_loader, another_loader])

    data = next(iter(joint_loader))
    assert "a" in data and "another" in data, "JointMemoryLoader did not yield expected memory data"


def test_joint_memory_loader_datagroup(
    a_dummy_table: ArrayMemoryTable, another_dummy_table: ArrayMemoryTable
):
    a_loader = MemoryLoader(
        memory_table=a_dummy_table,
        rollout_count=1,
        rollout_length=1,
        size_key="batch_size",
        data_group="a",
    )
    another_loader = MemoryLoader(
        memory_table=another_dummy_table,
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


def test_joint_memory_loader_nonunique_loaders_trigger_exception(a_dummy_table: ArrayMemoryTable):
    loader1 = MemoryLoader(
        memory_table=a_dummy_table,
        rollout_count=1,
        rollout_length=1,
        size_key="batch_size",
        data_group="a",
    )
    loader2 = MemoryLoader(
        memory_table=a_dummy_table,
        rollout_count=1,
        rollout_length=1,
        size_key="batch_size",
        data_group="a",
    )

    with pytest.raises(Exception, match="JointMemoryLoader"):
        joint_loader = JointMemoryLoader([loader1, loader2])  # noqa
