import numpy as np
import os
import torch
import argparse
import matplotlib.pyplot as plt
from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


def get_data_from_buffer(
        action_size: int,
        observation_size: int,
        memory_path: str,
        observation_key: str,
        max_memory_size: int,
        minimum_samples: int,
        use_terminal: bool = True,
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
        use_terminal_column=use_terminal,
        maxlen=max_memory_size,
        device=device,
    )
    table.restore(memory_path)
    print(f"the size of the table is: {table.size()}")
    seq_length = 50
    samples = table.sample(count=1, sequence_length=seq_length)
    observations = samples['observation'][observation_key]
    actions = samples['actions']
    while observations.shape[0] < minimum_samples:
        try:
            samples = table.sample(count=1, sequence_length=seq_length)
            observations = torch.cat((observations, samples['observation'][observation_key]), 0)
            actions = torch.cat((actions, samples['actions']), dim=0)
        except:
            print('problem with sampling')
    return observations.numpy(), actions.numpy()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer", type=str, default="/home/ali/data/biped/replay_buffer/")
    parser.add_argument("--path-to-mocap", type=str, default="/home/ali/data/biped/numpy/")
    parser.add_argument("--path-to-save", type=str, default="/home/ali/data/biped/figure/")
    parser.add_argument("--num-joints", type=int, default=17)
    parser.add_argument("--action-count", type=int, default=51)
    parser.add_argument("--obs-count", type=int, default=252)

    arg = parser.parse_args()

    path_to_buffer = arg.path_to_buffer
    path_to_mocap = arg.path_to_mocap
    path_to_save = arg.path_to_save
    action_count = arg.action_count
    obs_count = arg.obs_count

    obs_key = "features"
    buffer_observations, _ = get_data_from_buffer(
        action_size=action_count,
        observation_size=obs_count,
        memory_path=path_to_buffer,
        observation_key=obs_key,
        max_memory_size=500_000,
        minimum_samples=500
    )
    mocap_observations, _ = get_data_from_buffer(
        action_size=action_count,
        observation_size=182,
        memory_path=path_to_mocap,
        observation_key=obs_key,
        max_memory_size=500_000,
        minimum_samples=500,
        use_terminal=False,
    )

    print(f"size of buffer data: {buffer_observations.shape},\n"
          f"mocap data: {mocap_observations.shape}")

    n_joints = arg.num_joints
    joint_angle_idx = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(n_joints)]
    offset = n_joints * 4
    linear_velocity_idx = [[offset + 9 * k, offset + 9 * k + 1, offset + 9 * k + 2]
                           for k in range(n_joints)]
    position_idx = [[offset + 9 * k + 6, offset + 9 * k + 7, offset + 9 * k + 8]
                    for k in range(n_joints)]

    all_idx = {
        'joint_angle': joint_angle_idx,
        'linear_velocity': linear_velocity_idx,
        'linear_positions': position_idx,
    }

    font = {'size': 6}
    plt.rc('font', **font)

    for key, indices in all_idx.items():
        for j in range(3):
            for k in range(n_joints):
                plt.subplot(6, 3, k + 1)
                idx = indices[k][j]
                plt.subplots_adjust(left=0.1,
                                    bottom=0.1,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.4,
                                    hspace=0.4)
                plt.hist(buffer_observations[:, idx], alpha=0.5, color='red')
                plt.hist(mocap_observations[:, idx], alpha=0.5, color='blue')
            plt.savefig(os.path.join(path_to_save, f"{key}-{j}.png"), dpi=2000)
            plt.close()
