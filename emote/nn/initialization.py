import numpy as np

from torch import nn


def ortho_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, np.sqrt(2))
        nn.init.constant_(m.bias.data, 0.0)
