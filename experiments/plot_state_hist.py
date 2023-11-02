import numpy as np
import os
import torch
import argparse
import matplotlib.pyplot as plt
from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


def get_data_from_mocap(
        mocap_path: str
):
    observations = np.load(os.path.join(mocap_path, 'observations.npy'))
    actions = np.load(os.path.join(mocap_path, 'actions.npy'))
    return observations, actions


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
    parser.add_argument("--action-count", type=int, default=51)
    parser.add_argument("--obs-count", type=int, default=252)

    arg = parser.parse_args()

    path_to_buffer = arg.path_to_buffer
    path_to_mocap = arg.path_to_mocap
    path_to_save = arg.path_to_save
    action_count = arg.action_count
    obs_count = arg.obs_count

    obs_key = "features"
    buffer_observations, buffer_actions = get_data_from_buffer(
        action_size=action_count,
        observation_size=obs_count,
        memory_path=path_to_buffer,
        observation_key=obs_key,
        max_memory_size=300_000,
        minimum_samples=500
    )
    mocap_observations, mocap_actions = get_data_from_mocap(path_to_mocap)
    print(f"size of buffer data: {buffer_observations.shape},\n"
          f"mocap data: {mocap_observations.shape}")

    np.save(os.path.join(path_to_save, 'mocap_observations.npy'), mocap_observations)
    np.save(os.path.join(path_to_save, 'mocap_actions.npy'), mocap_actions)
    np.save(os.path.join(path_to_save, 'buffer_observations.npy'), buffer_observations)
    np.save(os.path.join(path_to_save, 'buffer_actions.npy'), buffer_actions)

    joint_angle_idx = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(17)]
    linear_velocity_idx = [[68 + 9 * k, 68 + 9 * k + 1, 68 + 9 * k + 2] for k in range(17)]
    angular_velocity_idx = [[68 + 9 * k + 3, 68 + 9 * k + 4, 68 + 9 * k + 5] for k in range(17)]
    position_idx = [[17 * 4 + 9 * k + 6, 17 * 4 + 9 * k + 7, 17 * 4 + 9 * k + 8] for k in range(17)]
    root_idx = [k + 239 for k in range(9)]

    all_idx = {
        'joint_angle': joint_angle_idx,
        'linear_velocity': linear_velocity_idx,
        'angular_velocity': angular_velocity_idx,
        'linear_positions': position_idx,
    }

    #font = {'size': 20}
    #plt.rc('font', **font)


    for key, indices in all_idx.items():
        for j in range(3):
            for k in range(17):
                plt.subplot(6, 3, k + 1)
                idx = indices[k][j]
                plt.subplots_adjust(left=0.1,
                                    bottom=0.1,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.4,
                                    hspace=0.4)
                plt.hist(buffer_observations[:, idx])
                plt.hist(mocap_observations[:, idx])
            plt.savefig(os.path.join(path_to_save, f"{key}-{j}.png"), dpi=2000)
            plt.close()

    for j in range(9):
        plt.subplot(3, 3, j + 1)
        idx = root_idx[j]
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        plt.hist(buffer_observations[:, idx])
        plt.hist(mocap_observations[:, idx])
    plt.savefig(os.path.join(path_to_save, f"root.png"), dpi=2000)
    plt.close()