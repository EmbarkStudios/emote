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

    return observations.numpy(), actions.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer", type=str, default="")
    parser.add_argument("--path-to-mocap", type=str, default="")
    parser.add_argument("--path-to-save", type=str, default="")
    parser.add_argument("--action-count", type=int, default=51)
    parser.add_argument("--obs-count", type=int, default=252)

    arg = parser.parse_args()

    action_count = arg.action_count
    obs_count = arg.obs_count
    obs_key = "features"

    if arg.path_to_buffer != "":
        observations, _ = get_data_from_buffer(
            action_size=action_count,
            observation_size=obs_count,
            memory_path=arg.path_to_buffer,
            observation_key=obs_key,
            max_memory_size=300_000,
        )
    elif arg.path_to_mocap != "":
        observations, _ = get_data_from_mocap(arg.path_to_mocap)
        observations = observations[200:]

    else:
        raise(IOError, "--path-to-buffer or --path-to-mocap must be provided")

    print(f"size of buffer data: {observations.shape},"
          f"mocap data: {observations.shape}")

    joint_position_indices = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(17)]

    linear_velocity_indices = [[68 + 9 * k, 68 + 9 * k + 1, 68 + 9 * k + 2] for k in range(17)]
    position_indices = [[68 + 9 * k + 6, 68 + 9 * k + 7, 68 + 9 * k + 8] for k in range(17)]
    joint_velocity_indices = [[68 + 9 * k + 3, 68 + 9 * k + 4, 68 + 9 * k + 5] for k in range(17)]

    dt = 1.0 / 30
    t = np.arange(0.0, 1.0, dt)
    len_data = t.shape[0]
    for j in range(17):
        for idx in range(3):
            plt.cla()
            plt.clf()
            plt.plot(t, observations[:len_data, joint_position_indices[j][idx]])
            plt.savefig(os.path.join(arg.path_to_save, f"joint_pos_{j}_{idx}.png"))
            plt.close()

            derivatives = (
                                 observations[1:len_data+1, joint_position_indices[j][idx]] -
                                 observations[0:len_data, joint_position_indices[j][idx]]
            ) / dt
            plt.cla()
            plt.clf()
            plt.plot(t, derivatives)
            plt.savefig(os.path.join(arg.path_to_save, f"derivatives_{j}_{idx}.png"))
            plt.close()

            plt.cla()
            plt.clf()
            plt.plot(t, observations[:len_data, joint_velocity_indices[j][idx]])
            plt.savefig(os.path.join(arg.path_to_save, f"joint_vel_{j}_{idx}.png"))
            plt.close()

            plt.cla()
            plt.clf()
            plt.plot(t, observations[:len_data, position_indices[j][idx]])
            plt.savefig(os.path.join(arg.path_to_save, f"position_{j}_{idx}.png"))
            plt.close()

            derivatives = (
                                  observations[1:len_data + 1, position_indices[j][idx]] -
                                  observations[0:len_data, position_indices[j][idx]]
                          ) / dt
            plt.cla()
            plt.clf()
            plt.plot(t, derivatives)
            plt.savefig(os.path.join(arg.path_to_save, f"position_derivatives_{j}_{idx}.png"))
            plt.close()

            plt.cla()
            plt.clf()
            plt.plot(t, observations[:len_data, linear_velocity_indices[j][idx]])
            plt.savefig(os.path.join(arg.path_to_save, f"linear_velocity_{j}_{idx}.png"))
            plt.close()
