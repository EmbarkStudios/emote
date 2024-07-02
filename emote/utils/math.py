# This file contains codes and texts that are copied from
# https://github.com/facebookresearch/mbrl-lib

import torch


def truncated_linear(min_x: float, max_x: float, min_y: float, max_y: float, x: float) -> float:
    """Truncated linear function.

    Implements the following function:

    \\[
    \\begin{cases}
        f1(x) = \\frac{min_y + (x - min_x)}{ (max_x - min_x) * (max_y - min_y)} \\\\
        f(x) = min(max_y, max(min_y, f1(x)))
    \\end{cases}
    \\]
    If max_x - min_x < 1e-10, then it behaves as the constant \\(f(x) = max_y\\)
    """
    if max_x - min_x < 1e-10:
        return max_y
    if x <= min_x:
        y: float = min_y
    else:
        dx = (x - min_x) / (max_x - min_x)
        dx = min(dx, 1.0)
        y = dx * (max_y - min_y) + min_y
    return y


def truncated_normal_(tensor: torch.Tensor, mean: float = 0, std: float = 1) -> torch.Tensor:
    """Samples from a truncated normal distribution in-place.

    Arguments:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        bound_violations = torch.sum(cond).item()
        if bound_violations == 0:
            break
        tensor[cond] = torch.normal(mean, std, size=(bound_violations,), device=tensor.device)
    return tensor
