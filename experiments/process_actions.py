import os

import numpy as np
import torch

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory.memory import MemoryLoader


memory_paths = ["/home/ali/data/bullcrab/replay_buffer/fast",
                "/home/ali/data/bullcrab/replay_buffer/slow"]

action_size = 5
input_shapes = {
    "features": {
        "shape": [246]
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
vae_spaces = MDPSpace(
    rewards=None,
    actions=BoxSpace(dtype=np.float32, shape=(action_size,)),
    state=DictSpace(state_spaces),
)

#import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#colors = ['blue', 'red']

for i in range(2):
    table = DictObsNStepTable(
        spaces=vae_spaces,
        use_terminal_column=True,
        maxlen=1_000_000,
        device=device,
    )
    restore_path = os.path.join(
        memory_paths[i], f"{data_group}_export"
    )
    print("restore_path: ", restore_path)
    table.restore(restore_path)
    print(f"the size of the table is: {table.size()}")

    data_loader = MemoryLoader(
        table,
        batch_size // rollout_length,
        rollout_length,
        "batch_size",
        data_group=data_group,
    )

    itr = iter(data_loader)
    data = next(itr)

    observation = data[data_group]['observation']['features']
    print(f"observation shape is {observation.shape}")
    print(f"observation at 208: {observation[:, 208]}")

    actions = data[data_group]['actions']
    print(actions.shape)
    print("mean: ", torch.mean(actions, dim=0))
    print("std: ", torch.std(actions, dim=0))

    #lat1 = actions[:, 2].squeeze().numpy()
    #lat2 = actions[:, 3].squeeze().numpy()
    #lat3 = actions[:, 4].squeeze().numpy()

    #ax.plot3D(lat1, lat2, lat3, '*', color=colors[i])

#plt.show()