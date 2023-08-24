import numpy as np

from torch import nn


def ortho_init_(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.constant_(m.bias.data, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def xavier_uniform_init_(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain)
        nn.init.constant_(m.bias.data, 0.0)
