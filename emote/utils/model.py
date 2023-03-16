# This file contains codes/text mostly restructured from the following github repository
# https://github.com/facebookresearch/mbrl-lib


from typing import Union

import numpy as np
import torch


def to_tensor(x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise ValueError("Input must be torch.Tensor or np.ndarray.")


def to_numpy(x: Union[torch.Tensor, np.ndarray]):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").numpy()
    if isinstance(x, np.ndarray):
        return x
    raise ValueError("Input must be torch.Tensor or np.ndarray.")
