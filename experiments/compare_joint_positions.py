import numpy as np
import os
import torch

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
    path_to_buffer = "/home/ali/data/biped/replay_buffer/tpos/rl_loader_export"
    path_to_mocap = "/home/ali/data/biped/numpy/fixed2"

    action_count = 51
    obs_count = 252
    obs_key = "features"
    num_samples = 100

    buffer_observations, buffer_actions = get_data_from_buffer(
        action_size=action_count,
        observation_size=obs_count,
        memory_path=path_to_buffer,
        observation_key=obs_key,
        max_memory_size=300_000,
        minimum_samples=num_samples
    )
    mocap_observations, mocap_actions = get_data_from_mocap(path_to_mocap)
    print(f"size of buffer data: {buffer_observations.shape},"
          f"mocap data: {mocap_observations.shape}")

    joint_angle_idx = [[4 * k, 4 * k + 1, 4 * k + 2] for k in range(17)]
    print(joint_angle_idx)

    #print('all joint values')
    #for group_idx in joint_angle_idx:
    #    for idx in group_idx:
    #        print(buffer_observations[0][idx])

    print('*'*20)
    print('lets print some joints from buffer')
    indices = [0, 1, 2, 4, 5, 6]
    for s in range(10):
        print(s, buffer_observations[s][indices])
    print('now from mocap')
    for s in range(10):
        print(s, mocap_observations[s][indices])
    print('*' * 20)

    #print('actions from buffer')
    #for s in range(10):
    #    print(s, buffer_actions[s][:10])


    ctr = 0
    joint_error = np.zeros((num_samples, 17 * 3))
    for joint_group in joint_angle_idx:
        for joint in joint_group:
            for s in range(num_samples):
                joint_error[s][ctr] = np.abs(mocap_observations[0][joint] - buffer_observations[s][joint])
            ctr += 1

    min_idx = np.argmin(np.sum(joint_error, axis=1))
    for idx in range(17*3):
        print(idx, joint_error[min_idx][idx])
