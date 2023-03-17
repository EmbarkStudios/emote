import torch


def to_numpy(x: torch.Tensor):
    return x.detach().to("cpu").numpy()
