import os
import numpy as np
import torch
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
        actions.pop()   # this is to remove the last action
        print("last action item is removed! ")
    rewards = [[0] for _ in range(num_samples-1)]

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


if __name__ == "__main__":
    path_to_mocap_data: str = "/home/ali/data/biped/numpy/walk1_subject1"
    preferred_device = torch.device('cpu')

    bc_actions = np.load(os.path.join(path_to_mocap_data, 'actions.npy'))
    bc_observations = np.load(os.path.join(path_to_mocap_data, 'observations.npy'))

    print(f"observation size: {bc_observations.shape}, "
          f"action size: {bc_actions.shape}")

    memory = create_table_from_numpy(bc_observations, bc_actions, "features", preferred_device)
    print(f"Table contains: {memory.size()} samples")
    memory.store("/home/ali/data/biped/replay_buffer/mocap/mocap")
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
