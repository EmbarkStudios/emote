"""

"""

from typing import List, Type

import numpy as np

from ..utils import MDPSpace
from .table import ArrayTable
from .column import Column, TagColumn, VirtualColumn
from .fifo_strategy import FifoEjectionStrategy
from .uniform_strategy import UniformSampleStrategy
from .storage import NextElementMapper, SyntheticDones
from .adaptors import DictObsAdaptor, TerminalAdaptor


def create_dict_table_from_columns(
    use_terminal_column: bool,
    max_size: int,
    columns: List[Type[Column]],
    obs_keys: List[str],
):
    if use_terminal_column:
        columns.append(
            TagColumn(
                name="terminal",
                shape=tuple(),
                dtype=np.float32,
            )
        )

    table = ArrayTable(
        columns,
        max_size,
        UniformSampleStrategy(),
        FifoEjectionStrategy(),
    )

    if use_terminal_column:
        table = TerminalAdaptor(table, "terminal", "masks")

    return DictObsAdaptor(table, obs_keys)


def create_dict_obs_table(
    spaces: MDPSpace,
    use_terminal_column: bool = False,
    max_size: int = 1_000_000,
):
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

    return create_dict_table_from_columns(
        use_terminal_column, max_size, columns, obs_keys
    )


def create_dict_obs_nstep_table(
    spaces: MDPSpace,
    use_terminal_column: bool,
    max_size: int,
):
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

    return create_dict_table_from_columns(
        use_terminal_column, max_size, spaces, columns, obs_keys
    )
