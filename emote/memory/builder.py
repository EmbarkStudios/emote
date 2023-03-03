"""

"""

import numpy as np
import torch

from ..utils import MDPSpace
from .adaptors import DictObsAdaptor, TerminalAdaptor
from .column import Column, TagColumn, VirtualColumn
from .fifo_strategy import FifoEjectionStrategy
from .storage import NextElementMapper, SyntheticDones
from .table import ArrayTable
from .uniform_strategy import UniformSampleStrategy


class DictTable(ArrayTable):
    def __init__(
        self,
        *,
        use_terminal_column: bool,
        obs_keys: list[str],
        columns: list[Column],
        maxlen: int,
        length_key="actions",
        device: torch.device,
    ):
        adaptors = [DictObsAdaptor(obs_keys)]
        if use_terminal_column:
            columns.append(
                TagColumn(
                    name="terminal",
                    shape=tuple(),
                    dtype=np.float32,
                )
            )
            adaptors.append(TerminalAdaptor("terminal", "masks"))

        super().__init__(
            columns=columns,
            maxlen=maxlen,
            sampler=UniformSampleStrategy(),
            ejector=FifoEjectionStrategy(),
            length_key=length_key,
            adaptors=adaptors,
            device=device,
        )


class DictObsTable(DictTable):
    """Create a memory suited for Reinforcement Learning Tasks with 1-Step Bellman
    Backup with a single bootstrap value, and using dictionary observations as network
    inputs.
    """

    def __init__(
        self,
        *,
        spaces: MDPSpace,
        use_terminal_column: bool = False,
        maxlen: int = 1_000_000,
        device: torch.device,
        dones_dtype=np.bool8,
        masks_dtype=np.float32,
    ):
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
                dtype=dones_dtype,
                shape=(1,),
                target_name="actions",
                mapper=SyntheticDones,
            ),
            VirtualColumn(
                name="masks",
                dtype=masks_dtype,
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

        super().__init__(
            use_terminal_column=use_terminal_column,
            maxlen=maxlen,
            columns=columns,
            obs_keys=obs_keys,
            device=device,
        )


class DictObsNStepTable(DictTable):
    """Create a memory suited for Reinforcement Learning Tasks with N-Step Bellman
    Backup with a single bootstrap value, and using dictionary observations as network
    inputs.
    """

    def __init__(
        self,
        *,
        spaces: MDPSpace,
        use_terminal_column: bool,
        maxlen: int,
        device: torch.device,
    ):
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

        super().__init__(
            use_terminal_column=use_terminal_column,
            maxlen=maxlen,
            columns=columns,
            obs_keys=obs_keys,
            device=device,
        )
