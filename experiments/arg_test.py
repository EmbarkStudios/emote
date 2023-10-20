import torch
from torch import Tensor


def func(key1: Tensor, key2: Tensor, key3: Tensor):
    print(f"key1: {key1}, "
          f"key2: {key2}, "
          f"key3: {key3}")


if __name__ == "__main__":
    dict1 = {
        'key1': torch.ones(1),
        'key2': 2 * torch.ones(2),
    }
    dict2 = {
        'key3': 3 * torch.ones(3)
    }
    func(**dict1, **dict2)

