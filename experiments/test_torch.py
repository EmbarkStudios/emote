import torch
from typing import List


def swap_indices_with_grad(tensor: torch.Tensor, left_idx: List[int], right_idx: List[int]):
    if len(left_idx) != len(right_idx):
        raise ValueError("left_idx and right_idx must have the same length")

    # Ensure the indices are within bounds
    data_size = tensor.size(1)
    if max(left_idx + right_idx) >= data_size:
        raise ValueError("Indices out of bounds")

    tensor.retain_grad()
    # Extract the values at left and right indices
    left_values = tensor[:, left_idx]
    right_values = tensor[:, right_idx]

    # Update the cloned tensor with swapped values
    # swapped_tensor = torch.zeros_like(tensor).retain_grad()
    tensor[:, left_idx] = right_values
    tensor[:, right_idx] = left_values

    return tensor

# Example usage
data = torch.randn(5, 10, requires_grad=True)
left_indices = [1, 3, 5]
right_indices = [2, 4, 6]

# Continue with your computations, and autograd will track gradients as usual
loss = 2.5 * data.sum()
loss.backward()

# Swap indices while maintaining gradients
swapped_data = swap_indices_with_grad(data, left_indices, right_indices)

# Access gradients for the original tensor
original_data_grad = data.grad
print(original_data_grad)

# Access gradients for the swapped tensor
swapped_data_grad = swapped_data.grad
print(swapped_data_grad)
