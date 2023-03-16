# This file contains codes/text mostly restructured from the following github repository
# https://github.com/facebookresearch/mbrl-lib

import numpy as np
import torch


#def to_tensor(x: np.ndarray, device: torch.device):
#    return torch.from_numpy(x).to(device)


def to_numpy(x: torch.Tensor):
    return x.detach().to("cpu").numpy()

