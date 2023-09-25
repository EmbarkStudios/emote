import os

import numpy as np
import torch

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace

memory_path = "/home/ali/data/komodo/replay_buffer/forward_skill/"

action_size = 28
input_shapes = {
    "features": {
        "shape": [272]
    }
}
data_group = "rl_loader"
device = torch.device('cpu')
batch_size = 1
rollout_length = 1

state_spaces = {
    k: BoxSpace(dtype=np.float32, shape=tuple(v["shape"]))
    for k, v in input_shapes.items()
}
vae_spaces = MDPSpace(
    rewards=None,
    actions=BoxSpace(dtype=np.float32, shape=(action_size,)),
    state=DictSpace(state_spaces),
)

table = DictObsNStepTable(
    spaces=vae_spaces,
    use_terminal_column=True,
    maxlen=1_000_000,
    device=device,
)
restore_path = os.path.join(
    memory_path, f"{data_group}_export"
)
print("restore_path: ", restore_path)
table.restore(restore_path)
print(f"the size of the table is: {table.size()}")

seq_length = 100
samples = table.sample(count=1, sequence_length=seq_length)
fp_idx = [149, 151, 155, 157]
fp = samples['observation']['features'][:, fp_idx]
print(fp)
