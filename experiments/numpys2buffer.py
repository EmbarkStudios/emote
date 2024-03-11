import os
import numpy as np
import torch
import argparse
from emote.memory import MemoryLoader
from experiments.numpy2buffer import create_table_from_numpy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer", type=str, default="/home/ali/data/buffer")
    parser.add_argument("--path-to-numpy", type=str, default="/home/ali/data/numpy/")

    parser.add_argument("--action-size", type=int, default=0)

    arg = parser.parse_args()
    preferred_device = torch.device('cpu')

    numpy_files = [f for f in os.listdir(arg.path_to_numpy) if f.endswith('.npy')]
    list_observations = []
    list_actions = []
    feature_size = 0
    for file_name in numpy_files:
        file_path = os.path.join(arg.path_to_numpy, file_name)
        obs = np.load(file_path)
        if feature_size == 0:
            feature_size = obs.shape[1]
        assert feature_size == obs.shape[1]
        num_samples = obs.shape[0]
        list_observations.append(obs)
        list_actions.append(np.zeros((num_samples, arg.action_size)))
    print(f"features: {feature_size}")

    memory = create_table_from_numpy(list_observations, list_actions, "features", preferred_device)
    print(f"Table contains: {memory.size()} samples")
    memory.store(arg.path_to_buffer)
    print(f"Memory saved successfully")

    batch_size = 4
    rollout_length = 10
    data_loader = MemoryLoader(
        memory,
        batch_size // rollout_length,
        rollout_length,
        "batch_size",
        data_group="rl_loader",
    )

    for _ in range(100):
        itr = iter(data_loader)
        data = next(itr)

    print("Memory test passed successfully")
