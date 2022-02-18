"""

"""

from typing import List, Type
from dataclasses import dataclass
import logging

import numpy as np

from ..utils import MDPSpace
from . import (
    ArrayTable,
    Column,
    FifoEjectionStrategy,
    UniformSampleStrategy,
    TagColumn,
    VirtualColumn,
)
from .storage import NextElementMapper, SyntheticDones
from .loading import fill_table_from_legacy_file
from .adaptors import DictObsAdaptor, TerminalAdaptor


BUILDER_LOGGER = logging.getLogger("BUILD")


@dataclass
class MemoryConfiguration:
    """Memory configuration object"""

    memory_min_size: int = 10_000  # deprecated
    memory_max_size: int = 1_000_000

    use_terminal_column: bool = False


def create_dict_table_from_columns(
    config: MemoryConfiguration,
    columns: List[Type[Column]],
    obs_keys: List[str],
):
    if config.use_terminal_column:
        columns.append(
            TagColumn(
                name="terminal",
                shape=tuple(),
                dtype=np.float32,
            )
        )

    table = ArrayTable(
        columns,
        config.memory_max_size,
        UniformSampleStrategy(),
        FifoEjectionStrategy(),
    )

    if config.use_terminal_column:
        table = TerminalAdaptor(table, "terminal", "masks")

    return DictObsAdaptor(table, obs_keys)


def create_dict_obs_memory(spaces: MDPSpace, config: MemoryConfiguration):
    """Create a memory suited for Reinforcement Learning Tasks with N-Step Bellman
    Backup with a single bootstrap value, and using dictionary observations as network
    inputs.
    """
    if spaces.rewards is not None:
        reward_column = Column(
            name="rewards",
            dtype=spaces.rewards.dtype,
            shape=spaces.rewards.shape,
        )
    else:
        reward_column = Column(name="rewards", dtype=np.float32, shape=(1,))

    columns = [
        Column(
            name="actions",
            dtype=spaces.actions.dtype,
            shape=spaces.actions.shape,
        ),
        VirtualColumn(
            name="dones",
            dtype=np.bool8,
            shape=(1,),
            target_name="actions",
            mapper=SyntheticDones,
        ),
        VirtualColumn(
            name="masks",
            dtype=np.float32,
            shape=(1,),
            target_name="actions",
            mapper=SyntheticDones.as_mask,
        ),
        reward_column,
    ]

    obs_keys = []
    for key, space in spaces.state.spaces.items():
        obs_keys.append(key)
        columns.extend(
            [
                Column(name=key, dtype=space.dtype, shape=space.shape),
                VirtualColumn(
                    name="next_" + key,
                    dtype=space.dtype,
                    shape=space.shape,
                    target_name=key,
                    mapper=NextElementMapper,
                ),
            ]
        )

    return create_dict_table_from_columns(config, columns, obs_keys)


def create_dict_obs_nstep_memory(spaces: MDPSpace, config: MemoryConfiguration):
    """Create a memory suited for Reinforcement Learning Tasks with N-Step Bellman
    Backup with a single bootstrap value, and using dictionary observations as network
    inputs.
    """
    if spaces.rewards is not None:
        reward_column = Column(
            name="rewards",
            dtype=spaces.rewards.dtype,
            shape=spaces.rewards.shape,
        )
    else:
        reward_column = Column(name="rewards", dtype=np.float32, shape=(1,))

    columns = [
        Column(
            name="actions",
            dtype=spaces.actions.dtype,
            shape=spaces.actions.shape,
        ),
        VirtualColumn(
            name="dones",
            dtype=np.bool8,
            shape=(1,),
            target_name="actions",
            mapper=SyntheticDones,
        ),
        VirtualColumn(
            name="masks",
            dtype=np.float32,
            shape=(1,),
            target_name="actions",
            mapper=SyntheticDones.as_mask,
        ),
        reward_column,
    ]

    obs_keys = []
    for key, space in spaces.state.spaces.items():
        obs_keys.append(key)
        columns.extend(
            [
                Column(name=key, dtype=space.dtype, shape=space.shape),
                VirtualColumn(
                    name="next_" + key,
                    dtype=space.dtype,
                    shape=space.shape,
                    target_name=key,
                    mapper=NextElementMapper.with_only_last,
                ),
            ]
        )

    return create_dict_table_from_columns(config, spaces, columns, obs_keys)


create_memory = create_dict_obs_memory
create_nstep_memory = create_dict_obs_nstep_memory


def create_intrinsic_memory(spaces: MDPSpace, config: MemoryConfiguration):
    """Create a memory suited for Reinforcement Learning Tasks with an Intrinsic
    Reward Column.
    """
    if spaces.rewards is not None:
        reward_column = Column(
            name="rewards",
            dtype=spaces.rewards.dtype,
            shape=spaces.rewards.shape,
        )
    else:
        reward_column = Column(name="rewards", dtype=np.float32, shape=(1,))

    columns = [
        Column(
            name="actions",
            dtype=spaces.actions.dtype,
            shape=spaces.actions.shape,
        ),
        VirtualColumn(
            name="dones",
            dtype=np.bool88,
            shape=(1,),
            target_name="actions",
            mapper=SyntheticDones,
        ),
        VirtualColumn(
            name="masks",
            dtype=np.float32,
            shape=(1,),
            target_name="actions",
            mapper=SyntheticDones.as_mask,
        ),
        reward_column,
        Column(name="intrinsic_rewards", dtype=np.float32, shape=(1,)),
    ]

    obs_keys = []
    for key, space in spaces.state.spaces.items():
        obs_keys.append(key)
        columns.extend(
            [
                Column(name=key, dtype=space.dtype, shape=space.shape),
                VirtualColumn(
                    name="next_" + key,
                    dtype=space.dtype,
                    shape=space.shape,
                    target_name=key,
                    mapper=NextElementMapper,
                ),
            ]
        )

    return create_dict_table_from_columns(config, spaces, columns, obs_keys)


def create_gail_reference_memory(
    spaces: MDPSpace, config: MemoryConfiguration, input_file: str
):
    """Create a memory suited for supervised-learning tasks on observations only,
    optionally loading from a file.  The file loading is very basic and loaded
    data is not expected to uphold the same variants as real memory at this
    point in time.
    """
    columns = [
        Column(
            name="obs",
            dtype=spaces.state.spaces["obs"].dtype,
            shape=spaces.state.spaces["obs"].shape,
        ),
    ]

    memory = DictObsAdaptor(
        ArrayTable(
            columns,
            config.memory_max_size,
            UniformSampleStrategy(),
            FifoEjectionStrategy(),
            length_key="obs",
        ),
        keys=["obs"],
        with_next=False,
    )

    if input_file:
        fill_table_from_legacy_file(memory, input_file, read_obs=True)

    return memory


def create_vd_reference_memory(
    spaces: MDPSpace, config: MemoryConfiguration, input_file: str
):
    """Create a memory suited for supervised-learning tasks on transitions without
    rewards, optionally loading from a file.  The file loading is very basic and
    loaded data is not expected to uphold the same variants as real memory at
    this point in time.

    """
    columns = [
        Column(
            name="obs",
            dtype=spaces.state.spaces["obs"].dtype,
            shape=spaces.state.spaces["obs"].shape,
        ),
        VirtualColumn(
            name="next_obs",
            target_name="obs",
            dtype=spaces.state.spaces["obs"].dtype,
            shape=spaces.state.spaces["obs"].shape,
            mapper=NextElementMapper,
        ),
        Column(
            name="actions",
            dtype=spaces.actions.dtype,
            shape=spaces.actions.shape,
        ),
    ]

    memory = DictObsAdaptor(
        ArrayTable(
            columns,
            config.memory_max_size,
            UniformSampleStrategy(),
            FifoEjectionStrategy(),
            length_key="actions",
        ),
        keys=["obs"],
    )

    if input_file:
        fill_table_from_legacy_file(
            memory, input_file, read_obs=True, read_actions=True
        )

    return memory


def create_bc_reference_memory(
    spaces: MDPSpace, config: MemoryConfiguration, input_file: str
):
    """Create a memory suited for supervised-learning tasks on state-actions pairs,
    optionally loading from a file.  The file loading is very basic and loaded
    data is not expected to uphold the same variants as real memory at this
    point in time.

    """
    columns = [
        Column(
            name="obs",
            dtype=spaces.state.spaces["obs"].dtype,
            shape=spaces.state.spaces["obs"].shape,
        ),
        Column(
            name="actions",
            dtype=spaces.actions.dtype,
            shape=spaces.actions.shape,
        ),
    ]

    memory = DictObsAdaptor(
        ArrayTable(
            columns,
            config.memory_max_size,
            UniformSampleStrategy(),
            FifoEjectionStrategy(),
            length_key="actions",
        ),
        keys=["obs"],
        with_next=False,
    )

    if input_file:
        fill_table_from_legacy_file(
            memory, input_file, read_obs=True, read_actions=True
        )

    return memory
