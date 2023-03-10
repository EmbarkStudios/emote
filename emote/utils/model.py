# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file contains codes/text mostly restructured from the following github repository
# https://github.com/facebookresearch/mbrl-lib


import numpy as np
import torch

from emote.typing import TensorType


def to_tensor(x: TensorType):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise ValueError("Input must be torch.Tensor or np.ndarray.")


def to_numpy(x: TensorType):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").numpy()
    if isinstance(x, np.ndarray):
        return x
    raise ValueError("Input must be torch.Tensor or np.ndarray.")
