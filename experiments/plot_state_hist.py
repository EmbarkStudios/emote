import numpy as np
import os
import torch

import matplotlib.pyplot as plt
from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


def get_data_from_mocap(
        mocap_path: str
):
    obs = np.load(os.path.join(mocap_path, 'observations.npy'))
    return obs


def get_data_from_buffer(
        action_size: int,
        observation_size: int,
        memory_path: str,
        observation_key: str,
        max_memory_size: int,
        minimum_samples: int,
):
    device = torch.device('cpu')
    spaces = MDPSpace(
        rewards=BoxSpace(dtype=np.float32, shape=(1,)),
        actions=BoxSpace(dtype=np.float32, shape=(action_size,)),
        state=DictSpace(
            {
                observation_key: BoxSpace(
                    dtype=np.float32,
                    shape=tuple([observation_size, ])
                )
            }
        ),
    )
    table = DictObsNStepTable(
        spaces=spaces,
        use_terminal_column=True,
        maxlen=max_memory_size,
        device=device,
    )
    table.restore(memory_path)
    print(f"the size of the table is: {table.size()}")
    seq_length = 10
    samples = table.sample(count=1, sequence_length=seq_length)['observation'][observation_key]
    while samples.shape[0] < minimum_samples:
        sample = table.sample(count=1, sequence_length=seq_length)
        samples = torch.cat((samples, sample['observation'][observation_key]), 0)
    return samples.numpy()


if __name__ == "__main__":
    path_to_buffer = "/home/ali/data/biped/replay_buffer/amp9/rl_batch_export"
    path_to_mocap = "/home/ali/data/biped/numpy/mocap"
    action_count = 51
    obs_count = 252
    obs_key = "features"
    buffer_obs = get_data_from_buffer(
        action_size=action_count,
        observation_size=obs_count,
        memory_path=path_to_buffer,
        observation_key=obs_key,
        max_memory_size=300_000,
        minimum_samples=4000
    )
    mocap_obs = get_data_from_mocap(path_to_mocap)
    print(f"size of buffer data: {buffer_obs.shape},"
          f"mocap data: {mocap_obs.shape}")

    for i in range(obs_count):
        plt.cla()
        plt.hist(buffer_obs[:, i])
        plt.hist(mocap_obs[:, i])
        plt.savefig(f"figures/obs{i}.png")
