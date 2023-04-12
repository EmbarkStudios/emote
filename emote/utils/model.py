import torch

from torch import nn


def to_numpy(x: torch.Tensor):
    return x.detach().to("cpu").numpy()


def normal_init(m: nn.Module):
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
