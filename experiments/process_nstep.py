import os

import numpy as np
import torch

from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from emote.memory.memory import MemoryLoader

from typing import List
from torch import Tensor


class FootstepReward:
    def __init__(
            self,
            footstep_indices: List[int],
            footstep_reference: Tensor,
            rollout_length: int,
            observation_key: str,
            data_group: str,
    ):
        super().__init__()
        self.data_group = data_group
        self.observation_key = observation_key
        self.rollout_length = rollout_length
        self._obs_key = observation_key
        self._footstep_indices = footstep_indices
        self._footstep_ref = torch.cat((footstep_reference[ :-1], footstep_reference[1:]), dim=1).to(torch.int8)
        print(self._footstep_ref)

    def begin_batch(
            self,
            observation: dict[str, Tensor],
            next_observation: dict[str, Tensor],
            rewards: Tensor
    ):
        obs = observation[self._obs_key]
        batch_size = obs.shape[0] // self.rollout_length
        obs = obs.reshape(batch_size , self.rollout_length, -1)
        reward = rewards.reshape(batch_size , self.rollout_length, -1)
        next_obs = next_observation[self._obs_key].unsqueeze(dim=1)
        obs = torch.cat((obs, next_obs), dim=1)

        footsteps = obs[:, :, self._footstep_indices].to(torch.int8)
        footsteps_transitions = torch.cat((footsteps[:, :-1], footsteps[:, 1:]), dim=2)

        footstep_rewards = torch.zeros_like(reward)
        for i in range(batch_size):
            for j in range(self.rollout_length):
                if any(torch.equal(footsteps_transitions[i, j], t) for t in self._footstep_ref):
                    footstep_rewards[i, j, 0] = 0.1
        print(footstep_rewards)
        total_rewards = reward + footstep_rewards
        return {self.data_group: {"rewards": total_rewards}}


memory_path = "/home/ali/data/komodo/replay_buffer/forward_skill/"

action_size = 28
feature_size = 272
batch_size = 10
rollout_length = 5
data_group = "rl_loader"
obs_key = "features"
device = torch.device('cpu')

input_shapes = {
    obs_key: {
        "shape": [feature_size]
    }
}

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
    maxlen=200_000,
    device=device,
)

restore_path = os.path.join(memory_path, f"{data_group}_export")
table.restore(restore_path)
print(f"the size of the table is: {table.size()}")

data_loader = MemoryLoader(
    table,
    batch_size,
    rollout_length,
    "batch_size",
    data_group=data_group,
)

footstep_ref = torch.asarray(
    [
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
    ]
)

rewarder = FootstepReward([149, 151, 155, 157], footstep_ref, rollout_length, obs_key, data_group)

itr = iter(data_loader)
data = next(itr)

observation = data[data_group]['observation']
next_observation = data[data_group]['next_observation']
reward = data[data_group]['rewards']

rewarder.begin_batch(observation, next_observation, reward)
