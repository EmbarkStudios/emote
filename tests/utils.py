import torch
from torch import nn
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsNStepTable
from tests.gym import DictGymWrapper


class FakeDataloader:
    def __init__(self, data_group: str, num_obs: int, batch_size: int):
        self.data_group = data_group
        self.num_obs = num_obs
        self.batch_size = batch_size

    def __len__(self):
        return 100

    def __iter__(self):
        while True:
            batch = {
                'observation': {'obs': torch.rand(self.batch_size, self.num_obs)},
            }
            yield {self.data_group: batch,
                   'batch_size': self.batch_size}


class RandomPolicy(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        rand_actions = 2 * (torch.rand(batch_size, self.action_dim) - 0.5)
        return rand_actions, 0


class MultiplierNN(nn.Module):
    def __init__(self, value: float, device: torch.device):
        super().__init__()
        self.value = value
        self.device = device

    def forward(self, x: torch.Tensor):
        return self.value * x

    def sample(self, x, rng):
        return self.forward(x)


def create_memory(
        env: DictGymWrapper,
        memory_size: int,
        len_rollout: int,
        batch_size: int,
        data_group: str,
        device: torch.device,
):
    """Creates memory and data_loader for the RL training

    Arguments:
        env (DictGymWrapper): the Gym-env wrapper
        memory_size (int): the maximum length of memory
        len_rollout (int): the rollout size for the NStepTable
        batch_size (int): batch size
        data_group (str): the data group for uploading the data
        device (torch.device): the device to upload the data
    Returns:
        (tuple[TableMemoryProxy, MemoryLoader]): A proxy for the memory and a dataloader

    """
    table = DictObsNStepTable(
        spaces=env.dict_space,
        use_terminal_column=False,
        maxlen=memory_size,
        device=device,
    )
    memory_proxy = TableMemoryProxy(table=table, use_terminal=False)
    data_loader = MemoryLoader(
        table=table,
        rollout_count=batch_size // len_rollout,
        rollout_length=len_rollout,
        size_key="batch_size",
        data_group=data_group,
    )
    return memory_proxy, data_loader
