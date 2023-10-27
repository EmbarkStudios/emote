import os

import numpy as np
import torch

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
print("restore_path: ", restore_path)
table.restore(restore_path)
print(f"the size of the table is: {table.size()}")

seq_length = 200

samples = table.sample(count=1, sequence_length=seq_length)
features = samples['observation']['features']
actions = samples['actions']
for i in range(5):
    samples = table.sample(count=1, sequence_length=seq_length)
    features_to_add = samples['observation']['features']
    actions_to_add = samples['actions']
    features = torch.cat((features, features_to_add), dim=0)
    actions = torch.cat((actions, actions_to_add), dim=0)

np_features = features.numpy()
np_actions = actions.numpy()

print(np_features.shape)
print(np_actions.shape)
np.save("actions.npy", np_actions)
np.save("features.npy", np_features)
