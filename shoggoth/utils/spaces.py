from typing import Tuple, Union, Dict
from dataclasses import dataclass

import numpy as np
import torch


class BoxSpace:
    dtype: Union[torch.dtype, np.dtype]
    shape: Tuple[int]


class DictSpace:
    spaces: Dict[str, BoxSpace]


@dataclass
class MDPSpace:
    rewards: BoxSpace
    actions: BoxSpace
    state: DictSpace
