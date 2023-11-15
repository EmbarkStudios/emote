import numpy as np
import torch
import argparse

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory import MemoryLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer", type=str, default="/home/ali/data/biped/replay_buffer/")
    parser.add_argument("--action-count", type=int, default=51)
    parser.add_argument("--obs-count", type=int, default=252)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--rollout", type=int, default=5)
    parser.add_argument("--use-terminal", action="store_true")

    arg = parser.parse_args()

    memory_path = arg.path_to_buffer
    action_size = arg.action_count

    if '.zip' in memory_path:
        memory_path = memory_path.replace('.zip', '')

    batch_size = arg.batch
    rollout_length = arg.rollout

    input_shapes = {
        "features": {
            "shape": [arg.obs_count]
        }
    }
    data_group = "rl_loader"
    device = torch.device('cpu')

    state_spaces = {
        k: BoxSpace(dtype=np.float32, shape=tuple(v["shape"]))
        for k, v in input_shapes.items()
    }
    spaces = MDPSpace(
        rewards=None,
        actions=BoxSpace(dtype=np.float32, shape=(action_size,)),
        state=DictSpace(state_spaces),
    )

    table = DictObsNStepTable(
        spaces=spaces,
        use_terminal_column=arg.use_terminal,
        maxlen=1_000_000,
        device=device,
    )

    table.restore(memory_path)
    print(f"the size of the table is: {table.size()}")

    data_loader = MemoryLoader(
        table,
        batch_size // rollout_length,
        rollout_length,
        "batch_size",
        data_group="rl_loader",
    )

    itr = iter(data_loader)

    data = next(itr)
    print('obs: ', data[data_group]['observation']['features'].shape)
    print('next_obs: ', data[data_group]['next_observation']['features'].shape)
    print('action: ', data[data_group]['actions'].shape)



