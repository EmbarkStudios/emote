import os

import numpy as np
import torch
from emote.memory.memory import MemoryLoader

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


experiment_path = '/home/ali/codes/erupt/logs/kh/genrl/bullcrab-no-vision'
trials = ['bullcrab-moveforward-fast-proprio-first-CONT',
          'bullcrab-moveforward-slow-proprio-first-CONT']
runs = ['0', '0']

action_size = 26
input_shapes = {
    "features": {
        "shape": [246]
    }
}
data_group = "rl_loader"
device = torch.device('cpu')
batch_size = 4000
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

memories = []
for trial, run in zip(trials, runs):
    table = DictObsNStepTable(
        spaces=vae_spaces,
        use_terminal_column=False,
        maxlen=1_000_000,
        device=device,
    )
    restore_path = os.path.join(
        experiment_path, trial, run, f"{data_group}_export"
    )
    print("restore_path: ", restore_path)
    table.restore(restore_path)
    print(f"the size of the table is: {table.size()}")
    memories.append(table)
data_loader = MemoryLoader(
    memories[0],
    batch_size // rollout_length,
    rollout_length,
    "batch_size",
    data_group=data_group,
)

itr = iter(data_loader)
data = next(itr)
print(data)