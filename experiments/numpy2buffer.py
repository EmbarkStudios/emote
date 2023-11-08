import os
import numpy as np
import torch
import argparse
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory.builder import DictObsNStepTable
from emote.memory import MemoryLoader


def create_table_from_numpy(
        observations: np.ndarray,
        actions: np.ndarray,
        observation_key: str,
        device: torch.device,
        use_terminal_masking: bool = False,
        max_size: int = 100_000
):
    num_samples = observations.shape[0]
    action_count = actions.shape[1]
    feature_count = observations.shape[1]

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

    features = [obs for obs in observations]
    actions = [action for action in actions]
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
    table.add_sequence(0, seq)

    return table


def reduce_samples(observations, actions, skip_sample=1):
    num_samples = observations.shape[0]
    idx = np.arange(0, num_samples, skip_sample + 1)
    return observations[idx], actions[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer", type=str, default="/home/ali/data/biped/replay_buffer/")
    parser.add_argument("--path-to-mocap", type=str, default="/home/ali/data/biped/numpy/")
    parser.add_argument("--action-count", type=int, default=51)
    parser.add_argument("--obs-count", type=int, default=252)
    parser.add_argument("--min-samples", type=int, default=4000)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--vision-size", type=int, default=0)

    arg = parser.parse_args()

    path_to_mocap_data = arg.path_to_mocap
    path_to_store_buffer = arg.path_to_buffer
    action_count = arg.action_count
    minimum_data = arg.min_samples
    skip_samples = arg.skip_samples

    preferred_device = torch.device('cpu')

    bc_actions = np.load(os.path.join(path_to_mocap_data, 'actions.npy'))
    bc_observations = np.load(os.path.join(path_to_mocap_data, 'observations.npy'))
    if action_count == 52:
        bc_actions = np.concatenate((bc_actions, np.zeros((bc_actions.shape[0], 1))), 1)

    if arg.vision_size > 0:
        bc_observations = np.concatenate(
            (
                bc_observations,
                np.zeros(
                    (bc_observations.shape[0], arg.vision_size)
                )
            ),
            1
        )

    print(f"observation size: {bc_observations.shape}")
    print(f"action size: {bc_actions.shape}")

    if skip_samples:
        bc_observations, bc_actions = reduce_samples(bc_observations, bc_actions, skip_sample=skip_samples)
        print(f"new observation size: {bc_observations.shape}, "
              f"new action size: {bc_actions.shape}")

    while bc_observations.shape[0] < minimum_data:
        bc_observations = np.concatenate((bc_observations, bc_observations), axis=0)
        bc_actions = np.concatenate((bc_actions, bc_actions), axis=0)
        print(f"observation size: {bc_observations.shape}, "
              f"action size: {bc_actions.shape}")

    memory = create_table_from_numpy(bc_observations, bc_actions, "features", preferred_device)
    print(f"Table contains: {memory.size()} samples")
    memory.store(path_to_store_buffer)
    print(f"Memory saved successfully")

    batch_size = 3
    rollout_length = 1
    data_loader = MemoryLoader(
        memory,
        batch_size // rollout_length,
        rollout_length,
        "batch_size",
        data_group="rl_loader",
    )

    itr = iter(data_loader)
    data = next(itr)
    # print(data)
