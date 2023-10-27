import numpy as np
import os
import torch

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
    path_to_buffer = "/home/ali/data/biped/replay_buffer/amp-joint-1/rl_batch_export"
    path_to_mocap = "/home/ali/data/biped/numpy/forward"
    path_to_save = "/home/ali/data/biped/artifacts/hist"
    action_count = 51
    obs_count = 252
    obs_key = "features"
    buffer_observations, buffer_actions = get_data_from_buffer(
        action_size=action_count,
        observation_size=obs_count,
        memory_path=path_to_buffer,
        observation_key=obs_key,
        max_memory_size=300_000,
        minimum_samples=4000
    )
    mocap_observations, mocap_actions = get_data_from_mocap(path_to_mocap)
    print(f"size of buffer data: {buffer_observations.shape},"
          f"mocap data: {mocap_observations.shape}")

    np.save(os.path.join(path_to_save, 'mocap_observations.npy'), mocap_observations)
    np.save(os.path.join(path_to_save, 'mocap_actions.npy'), mocap_actions)
    np.save(os.path.join(path_to_save, 'buffer_observations.npy'), buffer_observations)
    np.save(os.path.join(path_to_save, 'buffer_actions.npy'), buffer_actions)

    joint_angle_idx = [
        [
            [0, 1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14]
        ],
        [
            [16, 17, 18], [20, 21, 22], [24, 25, 26], [28, 29, 30]
        ],
        [
            [32, 33, 34], [36, 37, 38], [40, 41, 42], [44, 45, 46]
        ],
        [
            [48, 49, 50], [52, 53, 54], [56, 57, 58], [60, 61, 62]
        ],
        [
            [64, 65, 66]
        ]
    ]
    joint_error_idx = [[4 * k + 3] for k in range(17)]
    linear_velocity_idx = [[17 * 4 + 9 * k, 17 * 4 + 9 * k + 1, 17 * 4 + 9 * k + 2] for k in range(17)]
    angular_velocity_idx = [[17 * 4 + 9 * k + 3, 17 * 4 + 9 * k + 4, 17 * 4 + 9 * k + 5] for k in range(17)]
    position_idx = [[17 * 4 + 9 * k + 6, 17 * 4 + 9 * k + 7, 17 * 4 + 9 * k + 8] for k in range(17)]
    collision_idx = [[k + 221] for k in range(18)]
    root_idx = [k + 239 for k in range(9)]
    goal_idx = [k + 248 for k in range(4)]

    print(joint_angle_idx)
    #print(joint_error_idx)
    #print(linear_velocity_idx)
    #print(angular_velocity_idx)
    #print(position_idx)
    #print(collision_idx)
    #print(root_idx)
    #print(goal_idx)

    font = {'size': 3}
    plt.rc('font', **font)

    for group_idx, group in enumerate(joint_angle_idx):
        for i, list_idx in enumerate(group):
            for j, idx in enumerate(list_idx):
                plt.subplot(4, 3, j + i * 3 + 1)
                plt.subplots_adjust(left=0.1,
                                    bottom=0.1,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.4,
                                    hspace=0.4)
                plt.hist(buffer_observations[:, idx])
                plt.hist(mocap_observations[:, idx])
                plt.title(f"dimension {idx}")
        plt.savefig(os.path.join(path_to_save, f"game_joint_angles{group_idx}.png"), dpi=2000)
        plt.close()

    err

    for i, list_idx in enumerate(joint_angle_idx):
        for j, idx in enumerate(list_idx):
            plt.subplot(17, 3, j+i*3+1)
            plt.hist(mocap_observations[:, idx])
    plt.savefig(os.path.join(path_to_save, f"mocap_joint_angles.png"), dpi=2000)
    plt.close()

    errr
    for i, list_idx in enumerate(joint_error_idx):
        for j, idx in enumerate(list_idx):
            plt.subplot(17, 1, j+i+1)
            plt.hist(buffer_observations[:, idx])
            plt.hist(mocap_observations[:, idx])
    plt.savefig(os.path.join(path_to_save, f"joint_errors"), dpi=2000, format='png')
    plt.close()

    for i, list_idx in enumerate(linear_velocity_idx):
        for j, idx in enumerate(list_idx):
            plt.subplot(17, 3, j+i*3+1)
            plt.hist(buffer_observations[:, idx])
            plt.hist(mocap_observations[:, idx])
    plt.savefig(os.path.join(path_to_save, f"linear_velocities.png"), dpi=2000)
    plt.close()

    for i, list_idx in enumerate(angular_velocity_idx):
        for j, idx in enumerate(list_idx):
            plt.subplot(17, 3, j+i*3+1)
            plt.hist(buffer_observations[:, idx])
            plt.hist(mocap_observations[:, idx])
    plt.savefig(os.path.join(path_to_save, f"angular_velocities.png"), dpi=2000)
    plt.close()

    for i, list_idx in enumerate(position_idx):
        for j, idx in enumerate(list_idx):
            plt.subplot(17, 3, j+i*3+1)
            plt.hist(buffer_observations[:, idx])
            plt.hist(mocap_observations[:, idx])
    plt.savefig(os.path.join(path_to_save, f"position.png"), dpi=2000)
    plt.close()

    for i, list_idx in enumerate(collision_idx):
        for j, idx in enumerate(list_idx):
            plt.subplot(18, 1, j+i+1)
            plt.hist(buffer_observations[:, idx])
            plt.hist(mocap_observations[:, idx])
    plt.savefig(os.path.join(path_to_save, f"collision"), dpi=2000, format='png')
    plt.close()

    for i, idx in enumerate(root_idx):
        plt.subplot(3, 3, i+1)
        plt.hist(buffer_observations[:, idx])
        plt.hist(mocap_observations[:, idx])
    plt.savefig(os.path.join(path_to_save, f"root"), dpi=2000, format='png')
    plt.close()

    for i, idx in enumerate(goal_idx):
        plt.subplot(1, 4, i+1)
        plt.hist(buffer_observations[:, idx])
        plt.hist(mocap_observations[:, idx])
    plt.savefig(os.path.join(path_to_save, f"goal"), dpi=2000, format='png')
    plt.close()