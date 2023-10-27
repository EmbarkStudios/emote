import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace

memory_path = "/home/ali/data/biped/replay_buffer/amp5/"
data_group = "rl_batch"
action_size = 51
input_shapes = {
    "features": {
        "shape": [252]
    }
}
device = torch.device('cpu')
batch_size = 1
rollout_length = 1
memory_max_length = 300_000

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
    use_terminal_column=True,
    maxlen=memory_max_length,
    device=device,
)
restore_path = os.path.join(
    memory_path, f"{data_group}_export"
)
table.restore(restore_path)
print(f"the size of the table is: {table.size()}")

path_to_mocap_data: str = "/home/ali/data/biped/numpy/walk1_subject1"
bc_actions = np.load(os.path.join(path_to_mocap_data, 'actions.npy'))
bc_observations = np.load(os.path.join(path_to_mocap_data, 'observations.npy'))

print(f"observation size: {bc_observations.shape}, "
      f"action size: {bc_actions.shape}")

seq_length = 20

samples = table.sample(count=1, sequence_length=seq_length)
features = samples['observation']['features']
actions = samples['actions']
for i in range(100):
    samples = table.sample(count=1, sequence_length=seq_length)
    features_to_add = samples['observation']['features']
    actions_to_add = samples['actions']
    features = torch.cat((features, features_to_add), dim=0)
    actions = torch.cat((actions, actions_to_add), dim=0)
np_features = features.numpy()
np_actions = actions.numpy()

for i in range(features.shape[1]):
    plt.cla()
    plt.hist(bc_observations[:, i])
    plt.hist(np_features[:, i])
    plt.savefig(f"figures/plt{i}.png")
