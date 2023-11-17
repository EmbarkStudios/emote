import os
import numpy as np
import torch
import argparse
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory.builder import DictObsNStepTable
from emote.memory import MemoryLoader


def create_table_from_numpy(
        list_observations: list[np.ndarray],
        list_actions: list[np.ndarray],
        observation_key: str,
        device: torch.device,
        use_terminal_masking: bool = False,
        max_size: int = 100_000
):
    action_count = list_actions[0].shape[1]
    feature_count = list_observations[0].shape[1]

    spaces = MDPSpace(
        rewards=BoxSpace(dtype=np.float32, shape=(1,)),
        actions=BoxSpace(dtype=np.float32, shape=(action_count,)),
        state=DictSpace({observation_key: BoxSpace(dtype=np.float32, shape=tuple([feature_count, ]))}),
    )

    table = DictObsNStepTable(
        spaces=spaces,
        use_terminal_column=use_terminal_masking,
        maxlen=max_size,
        device=device,
    )

    num_sequences = len(list_observations)

    for k in range(num_sequences):

        num_samples = list_observations[k].shape[0]

        features = [obs for obs in list_observations[k]]
        actions = [action for action in list_actions[k]]
        if len(actions) == len(features):
            actions.pop()  # this is to remove the last action
            print("last action item is removed! ")
        rewards = [[0] for _ in range(num_samples - 1)]

        print(f"features: {len(features)}, "
              f"actions: {len(actions)},"
              f"rewards: {len(rewards)}")

        seq = {
            observation_key: features,
            "actions": actions,
            'rewards': rewards
        }
        table.add_sequence(k, seq)

    return table


def reduce_samples(observations, actions, skip_sample=1):
    num_samples = observations.shape[0]
    idx = np.arange(0, num_samples, skip_sample + 1)
    return observations[idx], actions[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer", type=str, default="/home/ali/data/biped/replay_buffer/")
    parser.add_argument("--path-to-mocap", type=str, default="/home/ali/data/biped/numpy/")
    parser.add_argument("--idx-start", nargs="+", type=int, default=[])
    parser.add_argument("--idx-end", nargs="+", type=int, default=[])
    parser.add_argument("--vision", action='store_true')
    parser.add_argument("--magic-force", action='store_true')
    # parser.add_argument("--skip-samples", type=int, default=0)

    arg = parser.parse_args()
    preferred_device = torch.device('cpu')

    bc_actions = np.load(os.path.join(arg.path_to_mocap, 'actions.npy'))
    bc_observations = np.load(os.path.join(arg.path_to_mocap, 'observations.npy'))

    if arg.magic_force:
        bc_actions = np.concatenate((bc_actions, np.zeros((bc_actions.shape[0], 1))), 1)

    if arg.vision:
        bc_observations = np.concatenate(
            (
                bc_observations,
                np.zeros(
                    (bc_observations.shape[0], 100)
                )
            ),
            1
        )

    print(f"observation size: {bc_observations.shape}")
    print(f"action size: {bc_actions.shape}")
    print('*'*20)

    list_observations = []
    list_actions = []

    assert len(arg.idx_start) == len(arg.idx_end)
    for idx_start, idx_end in zip(arg.idx_start, arg.idx_end):
        assert idx_end > idx_start
        list_observations.append(bc_observations[idx_start:idx_end])
        list_actions.append(bc_actions[idx_start:idx_end])

        print(f"observation size: {list_observations[-1].shape}")
        print(f"action size: {list_actions[-1].shape}")
        print('*' * 20)

    if not list_observations:
        list_observations.append(bc_observations)
        list_actions.append(bc_actions)

    # if arg.skip_samples:
    #    bc_observations, bc_actions = reduce_samples(bc_observations, bc_actions, skip_sample=arg.skip_samples)
    #    print(f"new observation size: {bc_observations.shape}, "
    #          f"new action size: {bc_actions.shape}")

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
