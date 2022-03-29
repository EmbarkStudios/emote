from torch import nn
import numpy as np


def ortho_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, np.sqrt(2))
        nn.init.constant_(m.bias.data, 0.0)