import numpy as np
import torch
from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory import MemoryLoader


vae_space = MDPSpace(
    rewards=BoxSpace(dtype=np.float32, shape=(1,)),
    actions=BoxSpace(dtype=np.float32, shape=(28,)),
    state=DictSpace({"features": BoxSpace(dtype=np.float32, shape=tuple([272,]))}),
)
table = DictObsNStepTable(
    spaces=vae_space,
    use_terminal_column=False,
    maxlen=1000,
    device=torch.device('cpu'),
)
for identity in range(100):
    seq = {
        'features': [np.ones(272) for _ in range(6)],
        'actions': [np.ones(28) for _ in range(5)],
        'rewards': [[0] for _ in range(5)]
    }
    table.add_sequence(identity, seq)
#table.store("memory_output")

print('table size is: ', table.size())

batch_size = 10
rollout_length = 1
data_loader = MemoryLoader(
    table,
    batch_size // rollout_length,
    rollout_length,
    "batch_size",
    data_group="rl_loader",
)

itr = iter(data_loader)
data = next(itr)
#print(data)
