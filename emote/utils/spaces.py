from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BoxSpace:
    dtype: torch.dtype | np.dtype
    shape: tuple[int]


@dataclass
class DictSpace:
    spaces: dict[str, BoxSpace]


@dataclass
class MDPSpace:
    rewards: BoxSpace
    actions: BoxSpace
    state: DictSpace
