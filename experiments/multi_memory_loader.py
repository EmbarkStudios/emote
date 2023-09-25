from typing import Dict, List, Optional
import numpy as np
import torch
from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.utils.timed_call import TimedBlock
from emote.memory.table import Table


def cat_dict_tensors(dict1, dict2):
    for key, value in dict1.items():
        if isinstance(value, dict):
            cat_dict_tensors(value, dict2[key])
        else:
            dict1[key] = torch.cat((value, dict2[key]), dim=0)
    if not dict1:
        dict1.update(dict2)


class MultiMemoryLoader:
    def __init__(
        self,
        tables: List[Table],
        rollout_count: int,
        rollout_length: int,
        size_key: str,
        data_group: str = "default",
    ):
        self.data_group = data_group
        self.tables = tables
        self.num_tables = len(tables)
        self.rollout_count = rollout_count
        self.rollout_length = rollout_length
        self.size_key = size_key
        self.timer = TimedBlock()
        self.table_indices = list(range(self.num_tables))

    def is_ready(self):
        """True if the data loader has enough data to start providing data"""
        return self.tables[0].size() >= (self.rollout_count * self.rollout_length)

    def __iter__(self):
        if not self.is_ready():
            raise Exception(
                "Data loader does not have enough data.\
                 Check `is_ready()` before trying to iterate over data."
            )

        while True:
            with self.timer:
                table_weights = np.array([self.tables[i].size() for i in range(self.num_tables)], dtype=np.float32)
                table_weights /= np.sum(table_weights)
                random_indices = np.random.choice(self.table_indices, size=self.rollout_count, p=table_weights)
                data = {}
                for i in range(self.num_tables):
                    rollout_count = np.sum(random_indices == i)
                    data_to_cat = self.tables[i].sample(rollout_count, self.rollout_length)
                    cat_dict_tensors(data, data_to_cat)
            data[self.size_key] = self.rollout_count * self.rollout_length
            yield {self.data_group: data, self.size_key: data[self.size_key]}


feature_size = 4
action_size = 2

device = torch.device('cpu')
vae_space = MDPSpace(
    rewards=BoxSpace(dtype=np.float32, shape=(1,)),
    actions=BoxSpace(dtype=np.float32, shape=(action_size,)),
    state=DictSpace({"features": BoxSpace(dtype=np.float32, shape=tuple([feature_size,]))}),
)
table1 = DictObsNStepTable(spaces=vae_space, use_terminal_column=False, maxlen=1000, device=device)
table2 = DictObsNStepTable(spaces=vae_space, use_terminal_column=False, maxlen=1000, device=device)
table3 = DictObsNStepTable(spaces=vae_space, use_terminal_column=False, maxlen=1000, device=device)
seq1 = {
    'features': [np.ones(feature_size) for _ in range(101)],
    'actions': [np.ones(action_size) for _ in range(100)],
    'rewards': [[1] for _ in range(100)]
}
seq2 = {
    'features': [np.zeros(feature_size) for _ in range(101)],
    'actions': [np.zeros(action_size) for _ in range(100)],
    'rewards': [[0] for _ in range(100)]
}
seq3 = {
    'features': [np.random.rand(feature_size) for _ in range(1001)],
    'actions': [np.random.rand(action_size) for _ in range(1000)],
    'rewards': [[np.random.rand(1)] for _ in range(1000)]
}
table1.add_sequence(0, seq1)
table2.add_sequence(0, seq2)
table3.add_sequence(0, seq3)

loader = MultiMemoryLoader([table1, table2, table3], 10, 1, "batch_size")

itr = iter(loader)
for k in range(10):
    batch = next(itr)
    #print(batch['default']['observation']['features'])
    print(batch['default']['actions'])
    print('*'*40)
