import numpy as np
import os
import torch

import matplotlib.pyplot as plt
from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory.memory import MemoryLoader


def get_data_from_buffer(
        action_size: int,
        observation_size: int,
        memory_path: str,
        observation_key: str,
        max_memory_size: int,
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
    batch_size = 4000
    rollout_length = 1
    data_group = "rl_data"
    data_loader = MemoryLoader(
        table,
        batch_size,
        rollout_length,
        "batch_size",
        data_group=data_group,
    )
    itr = iter(data_loader)
    data = next(itr)
    obs = data[data_group]['observation'][observation_key]
    print(obs.shape)
    joint_angle_idx = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(17)]
    print(joint_angle_idx)

    for list_idx in joint_angle_idx:
        for idx in list_idx:
            min_value = torch.min(obs[:, idx])
            max_value = torch.max(obs[:, idx])
            if min_value < -3.0 or max_value > 3.0:
                print(idx, min_value, max_value)


if __name__ == "__main__":
    path_to_buffer = "/home/ali/data/biped/replay_buffer/amp-joint-1/rl_batch_export"
    action_count = 51
    obs_count = 252
    obs_key = "features"
    get_data_from_buffer(
        action_size=action_count,
        observation_size=obs_count,
        memory_path=path_to_buffer,
        observation_key=obs_key,
        max_memory_size=300_000,
    )
