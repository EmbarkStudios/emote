import os
import numpy as np
import torch
import argparse
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory.builder import DictObsNStepTable
import matplotlib.pyplot as plt


def create_and_load_table(
        action_size: int,
        observation_size: int,
        use_terminal: bool,
        memory_max_size: int,
        memory_path: str,
        observation_key: str = "features"
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
        maxlen=memory_max_size,
        device=device,
    )
    table.restore(memory_path)
    return table


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer-1", type=str, default="/home/ali/data/biped/numpy/")
    parser.add_argument("--path-to-buffer-2", type=str, default="/home/ali/data/biped/numpy/")
    parser.add_argument("--path-to-results", type=str, default="/home/ali/data/biped/numpy/")
    parser.add_argument("--action-count", type=int, default=36)
    parser.add_argument("--observation-count", type=int, default=182)
    parser.add_argument("--number-of-joints", type=int, default=12)

    arg = parser.parse_args()

    n_joints = arg.number_of_joints
    joint_angle_idx = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(n_joints)]

    offset = n_joints * 4
    linear_velocity_idx = [[offset + 9 * k, offset + 9 * k + 1, offset + 9 * k + 2]
                           for k in range(n_joints)]
    angular_velocity_idx = [[offset + 9 * k + 3, offset + 9 * k + 4, offset + 9 * k + 5]
                            for k in range(n_joints)]
    position_idx = [[offset + 9 * k + 6, offset + 9 * k + 7, offset + 9 * k + 8]
                    for k in range(n_joints)]

    table_1 = create_and_load_table(
        action_size=arg.action_count,
        observation_size=arg.observation_count,
        use_terminal=False,
        memory_max_size=500000,
        memory_path=arg.path_to_buffer_1
    )

    table_2 = create_and_load_table(
        action_size=arg.action_count,
        observation_size=arg.observation_count,
        use_terminal=False,
        memory_max_size=500000,
        memory_path=arg.path_to_buffer_2
    )

    seq_length = 1296
    succeeded = False
    samples_1 = {}
    samples_2 = {}
    while not succeeded:
        try:
            samples_1 = table_1.sample(count=1, sequence_length=seq_length)
            samples_2 = table_2.sample(count=1, sequence_length=seq_length)
            succeeded = True
        except:
            print("")

    for plot_figure in range(2):
        plt_idx = 1
        for joint in range(6):
            for k in range(3):
                plt.subplot(6, 3, plt_idx)
                plt.hist(
                    samples_1["observation"]["features"][:, position_idx[joint + 6 * plot_figure][k]],
                    alpha=0.5,
                    color='red'
                )
                plt.hist(
                    samples_2["observation"]["features"][:, position_idx[joint + 6 * plot_figure][k]],
                    alpha=0.5,
                    color='blue'
                )
                plt_idx += 1

        filename = f"positions_{plot_figure}.png"
        filename = os.path.join(arg.path_to_results, filename)
        plt.savefig(filename)
        plt.close()

    for plot_figure in range(2):
        plt_idx = 1
        for joint in range(6):
            for k in range(3):
                plt.subplot(6, 3, plt_idx)
                plt.hist(
                    samples_1["observation"]["features"][:, joint_angle_idx[joint + 6 * plot_figure][k]],
                    alpha=0.5,
                    color='red'
                )
                plt.hist(
                    samples_2["observation"]["features"][:, joint_angle_idx[joint + 6 * plot_figure][k]],
                    alpha=0.5,
                    color='blue'
                )
                plt_idx += 1

        filename = f"joints_{plot_figure}.png"
        filename = os.path.join(arg.path_to_results, filename)
        plt.savefig(filename)
        plt.close()

    for plot_figure in range(2):
        plt_idx = 1
        for joint in range(6):
            for k in range(3):
                plt.subplot(6, 3, plt_idx)
                plt.hist(
                    samples_1["observation"]["features"][:, linear_velocity_idx[joint + 6 * plot_figure][k]],
                    alpha=0.5,
                    color='red'
                )
                plt.hist(
                    samples_2["observation"]["features"][:, linear_velocity_idx[joint + 6 * plot_figure][k]],
                    alpha=0.5,
                    color='blue'
                )
                plt_idx += 1

        filename = f"velocity_{plot_figure}.png"
        filename = os.path.join(arg.path_to_results, filename)
        plt.savefig(filename)
        plt.close()