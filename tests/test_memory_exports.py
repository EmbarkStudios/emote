import os
import stat

import numpy as np
import pytest

from emote.memory.column import Column, VirtualColumn
from emote.memory.fifo_strategy import FifoEjectionStrategy
from emote.memory.storage import SyntheticDones
from emote.memory.table import ArrayTable, TableSerializationVersion
from emote.memory.uniform_strategy import UniformSampleStrategy


@pytest.fixture
def memory():
    spec = [
        Column(name="observation", dtype=np.dtype("float32"), shape=tuple()),
        Column(name="reward", dtype=np.float32, shape=tuple()),
        VirtualColumn(
            name="dones",
            dtype=bool,
            shape=(1,),
            target_name="reward",
            mapper=SyntheticDones,
        ),
        VirtualColumn(
            name="masks",
            dtype=np.float32,
            shape=(1,),
            target_name="reward",
            mapper=SyntheticDones.as_mask,
        ),
    ]

    memory = ArrayTable(
        columns=spec,
        maxlen=10_000,
        sampler=UniformSampleStrategy(),
        ejector=FifoEjectionStrategy(),
        length_key="reward",
        device="cpu",
    )

    return memory


def test_export_base(memory, tmpdir):
    for ii in range(0, 1000):
        memory.add_sequence(ii, dict(observation=[1, 2, 3, 4, 5], reward=[1, 2, 3, 4]))

    original_observation_data = memory._data["observation"]
    original_reward_data = memory._data["reward"]

    export_file = os.path.join(tmpdir, "export")
    res_file = os.path.join(tmpdir, "export.zip")

    memory.store(export_file)

    assert os.path.exists(res_file), "written file must be exist"
    assert os.stat(res_file).st_size > 10_000, "should contain at least 10 000 bytes"

    st = os.stat(res_file)
    # has to be readable by the whole world
    required_perms = stat.S_IRUSR | stat.S_IROTH | stat.S_IRGRP
    assert (
        st.st_mode & required_perms
    ) == required_perms, "file should be readable by everyone"

    memory.restore(export_file)

    loaded_observation_data = memory._data["observation"]
    loaded_reward_data = memory._data["reward"]

    for identity in range(0, 1000):
        assert np.all(
            original_observation_data[identity] == loaded_observation_data[identity]
        ), "observation data should be the same"
        assert np.all(
            original_reward_data[identity] == loaded_reward_data[identity]
        ), "reward data should be the same"

    for ii in range(2000, 5000):
        memory.add_sequence(ii, dict(observation=[1, 2, 3, 4, 5], reward=[1, 2, 3, 4]))


def test_export_legacy(memory, tmpdir):
    for ii in range(0, 1000):
        memory.add_sequence(ii, dict(observation=[1, 2, 3, 4, 5], reward=[1, 2, 3, 4]))

    export_file = os.path.join(tmpdir, "export")
    res_file = os.path.join(tmpdir, "export.zip")

    memory.store(export_file, version=TableSerializationVersion.Legacy)

    assert os.path.exists(res_file), "written file must be exist"
    assert os.stat(res_file).st_size > 10_000, "should contain at least 10 000 bytes"

    st = os.stat(res_file)
    # has to be readable by the whole world
    required_perms = stat.S_IRUSR | stat.S_IROTH | stat.S_IRGRP
    assert (
        st.st_mode & required_perms
    ) == required_perms, "file should be readable by everyone"

    memory.restore(export_file)  # remove .zip


def test_import_v1(memory):
    for ii in range(0, 100):
        memory.add_sequence(ii, dict(observation=[1, 2, 3, 4, 5], reward=[1, 2, 3, 4]))

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    export_file = os.path.join(data_dir, "export-v1")

    original_observation_data = memory._data["observation"]
    original_reward_data = memory._data["reward"]

    memory.restore(export_file)

    loaded_observation_data = memory._data["observation"]
    loaded_reward_data = memory._data["reward"]

    for identity in range(0, 100):
        assert np.all(
            original_observation_data[identity] == loaded_observation_data[identity]
        ), "observation data should be the same"
        assert np.all(
            original_reward_data[identity] == loaded_reward_data[identity]
        ), "reward data should be the same"
