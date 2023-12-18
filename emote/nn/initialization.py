import numpy as np
import torch

from torch import nn


def ortho_init_(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def xavier_uniform_init_(m, gain):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain)
        nn.init.constant_(m.bias, 0.0)


def normal_init_(m: nn.Module):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, std=1e-3)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def trunc_normal_init_(m: nn.Module, std: float = 0.02):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def constant_init_(m: nn.Module, val):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, val)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
