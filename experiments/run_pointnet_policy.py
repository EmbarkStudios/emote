import numpy as np
import torch
import argparse
from emote.nn.layers import PointNetEncoder

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory.memory import MemoryLoader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-buffer", type=str, default="/home/ali/data/biped/replay_buffer/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--action-dim", type=int, default=22)
    parser.add_argument("--observation-dim", type=int, default=225)
    parser.add_argument("--point-count", type=int, default=300)
    parser.add_argument("--point-dim", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=5)
    parser.add_argument("--memory-size", type=int, default=500000)
    parser.add_argument("--use-terminal", action='store_true')

    arg = parser.parse_args()

    memory_path = arg.path_to_buffer.replace('.zip', '')
    device = torch.device(arg.device)

    obs_key = 'features'
    pc_key = 'point_cloud'

    action_size = arg.action_dim
    input_shapes = {
        obs_key: {
            "shape": [arg.observation_dim]
        },
        pc_key: {
            "shape": [arg.point_count * arg.point_dim]
        }
    }

    state_spaces = {
        k: BoxSpace(dtype=np.float32, shape=tuple(v["shape"]))
        for k, v in input_shapes.items()
    }

    input_spaces = MDPSpace(
        rewards=None,
        actions=BoxSpace(dtype=np.float32, shape=(action_size,)),
        state=DictSpace(state_spaces),
    )

    table = DictObsNStepTable(
        spaces=input_spaces,
        use_terminal_column=arg.use_terminal,
        maxlen=arg.memory_size,
        device=device,
    )

    table.restore(memory_path)
    print(f"the size of the table is: {table.size()}")
    data_loader = MemoryLoader(
        table,
        1,
        arg.episode_length,
        "batch_size",
        data_group="group",
    )
    print('*' * 30)

    itr = iter(data_loader)
    data = next(itr)
    obs = data['group']['observation'][obs_key]
    pc = data['group']['observation'][pc_key]
    actions = data['group']['actions']

    print(f"observation shape {obs.shape}")
    print(f"point cloud shape {pc.shape}")
    print(f"action shape {actions.shape}")


