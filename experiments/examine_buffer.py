import numpy as np
import torch

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory import MemoryLoader

memory_path = "/home/ali/data/biped/replay_buffer/forward/0/rl_loader_export"

action_size = 51
input_shapes = {
    "features": {
        "shape": [254]
    }
}
data_group = "rl_loader"
device = torch.device('cpu')
batch_size = 10
rollout_length = 1

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
    use_terminal_column=True,
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
for k in range(100000):
    data = next(itr)
    assert data[data_group]['observation']['features'].shape[1] == 254
    # print(data[data_group]['observation']['features'].shape)


