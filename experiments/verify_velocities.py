import numpy as np
import os
import torch
import argparse
from matplotlib import pyplot as plt

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
    parser.add_argument("--action-count", type=int, default=51)
    parser.add_argument("--obs-count", type=int, default=252)

    arg = parser.parse_args()

    action_count = arg.action_count
    obs_count = arg.obs_count
    obs_key = "features"
    num_samples = 100

    mocap_observations, buffer_actions = get_data_from_buffer(
        action_size=action_count,
        observation_size=obs_count,
        memory_path=arg.path_to_buffer,
        observation_key=obs_key,
        max_memory_size=300_000,
        minimum_samples=num_samples
    )
    buffer_observations, mocap_actions = get_data_from_mocap(arg.path_to_mocap)
    print(f"size of buffer data: {buffer_observations.shape},"
          f"mocap data: {mocap_observations.shape}")

    lin_vel_idx = [[68 + 9 * k, 68 + 9 * k + 1, 68 + 9 * k + 2] for k in range(17)]
    pos_idx = [[68 + 9 * k + 6, 68 + 9 * k + 7, 68 + 9 * k + 8] for k in range(17)]

    #for vel_idx, pos_idx in zip(lin_vel_idx, pos_idx):
    #    print('{')
    #    print(f"\t\"start\": {vel_idx[0]},")
    #    print(f"\t\"end\": {vel_idx[2]}")
    #    print('},')
    #    print('{')
    #    print(f"\t\"start\": {pos_idx[0]},")
    #    print(f"\t\"end\": {pos_idx[2]}")
    #    print('},')


    print(lin_vel_idx)
    print(pos_idx)
    joint_angle_idx = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(17)]
    joint_velocity_idx = [[68 + 9 * k + 3, 68 + 9 * k + 4, 68 + 9 * k + 5] for k in range(17)]
    print('joint angle indices: ', joint_angle_idx)
    print('joint velocity indices: ', joint_velocity_idx)
    err
    dt = 1.0 / 30
    t = np.arange(0.0, 3.0, dt)
    len_data = t.shape[0]
    for j in range(17):
        for idx in range(3):
            plt.cla()
            plt.clf()
            plt.plot(t, buffer_observations[:len_data, joint_angle_idx[j][idx]])
            plt.savefig(f"figures/joint_pos_{j}_{idx}.png")
            plt.close()

            velocities = (
                    buffer_observations[1:len_data+1, joint_angle_idx[j][idx]] -
                    buffer_observations[0:len_data, joint_angle_idx[j][idx]]
            ) / dt
            plt.cla()
            plt.clf()
            plt.plot(t, velocities)
            plt.savefig(f"figures/derivatives_{j}_{idx}.png")
            plt.close()

            plt.cla()
            plt.clf()
            plt.plot(t, buffer_observations[:len_data, joint_velocity_idx[j][idx]])
            plt.savefig(f"figures/joint_vel_{j}_{idx}.png")
            plt.close()



    """
    dt = 1.0 / 15
    for s in range(10):
        for pos_idx, vel_idx in zip([0, 1, 2], [80, 81, 82]):
            print(f"{vel_idx}: velocity1: {buffer_observations[s][vel_idx]}")
            velocity = (buffer_observations[s+1][pos_idx] - buffer_observations[s][pos_idx]) / dt
            print(f"{pos_idx}: velocity2: {velocity}")
    """