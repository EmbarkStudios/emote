from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import torch


@dataclass
class BoxSpace:
    dtype: torch.dtype | np.dtype
    shape: Tuple[int]


@dataclass
class DictSpace:
    spaces: Dict[str, BoxSpace]


@dataclass
class MDPSpace:
    rewards: BoxSpace
    actions: BoxSpace
    state: DictSpace
