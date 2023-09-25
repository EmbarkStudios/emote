import torch

def concatenate_tensors(dict1, dict2):
    for key, value in dict1.items():
        if isinstance(value, dict):
            concatenate_tensors(value, dict2[key])
        else:
            dict1[key] = torch.cat((value, dict2[key]), dim=0)

# Example usage:
data1 = {
    'tensor1': torch.tensor([1, 2]),
    'tensor2': torch.tensor([3, 4]),
    'nested': {
        'nested_tensor1': torch.tensor([5, 6]),
        'nested_tensor2': torch.tensor([7, 8]),
    }
}

data2 = {
    'tensor1': torch.tensor([9, 10]),
    'tensor2': torch.tensor([11, 12]),
    'nested': {
        'nested_tensor1': torch.tensor([13, 14]),
        'nested_tensor2': torch.tensor([15, 16]),
    }
}

concatenate_tensors(data1, data2)
print(data1)

