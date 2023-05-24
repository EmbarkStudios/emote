import torch

from gymnasium.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper, HitTheMiddle, SimpleGymCollector
from torch import nn

from emote import Trainer
from emote.callback import BatchCallback
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsTable
from emote.sac import FeatureAgentProxy
from emote.trainer import TrainingShutdownException


class RandomPolicy(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        rand_actions = 2 * (torch.rand(batch_size, self.action_dim) - 0.5)
        return rand_actions, 0


class HitTheMiddleDataInspector(BatchCallback):
    def __init__(
        self,
        num_bp: int,
        device: torch.device,
        data_group: str = "default",
    ):
        super().__init__()
        self.data_group = data_group
        self.cycle = num_bp
        self.device = device

    def begin_batch(self, *args, **kwargs):
        obs, next_obs, action, reward = self.get_batch(*args, **kwargs)
        batch_size = obs.shape[0]
        sim_next_obs, sim_reward = self.simulate_hit_the_middle(action, obs)
        for i in range(batch_size):
            obs_err = torch.mean(torch.abs(sim_next_obs[i] - next_obs[i])).detach()
            reward_err = torch.mean(torch.abs(sim_reward[i] - reward[i])).detach()
            if obs_err > 0.001 or reward_err > 0.001:
                message = f"""
                    obs_err: {obs_err}, reward_err: {reward_err}
                    obs: {obs[i]}, action: {action[i]}
                    reward: {reward[i]}, sim_reward: {sim_reward[i]}
                    next_obs: {next_obs[i]}, sim_next_obs: {sim_next_obs[i]}
                """
                raise ValueError(
                    f"Loaded values for obs/reward does not match the calculated ones {message}"
                )

    def simulate_hit_the_middle(self, action, obs):
        batch_size = action.shape[0]
        next_reward = torch.zeros(batch_size, 1)
        next_obs = torch.zeros(batch_size, 2)
        for i in range(batch_size):
            pos, vel = obs[i, 0].clone(), obs[i, 1].clone()
            vel += action[i, 0]
            pos += vel

            if pos > 10.0:
                pos = 10.0
                vel *= -1.0
            elif pos < -10.0:
                pos = -10.0
                vel *= -1.0
            next_reward[i] = -(pos**2)
            next_obs[i, :] = torch.Tensor([pos, vel])

        return next_obs.to(self.device), next_reward.to(self.device)

    def end_cycle(self):
        raise TrainingShutdownException()

    def get_batch(self, observation, next_observation, actions, rewards):
        return observation["obs"], next_observation["obs"], actions, rewards


def test_gym_collector():
    device = torch.device("cpu")
    batch_size = 5
    rollout_length = 1  # The test only works for rollout_length = 1
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(
        spaces=env.dict_space,
        use_terminal_column=False,
        maxlen=1000000,
        device=device,
    )
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(
        table=table,
        rollout_count=batch_size // rollout_length,
        rollout_length=rollout_length,
        size_key="batch_size",
    )

    policy = RandomPolicy(action_dim=1)
    agent_proxy = FeatureAgentProxy(policy, device)
    callbacks = [
        HitTheMiddleDataInspector(500, device),
        SimpleGymCollector(
            env, agent_proxy, memory_proxy, warmup_steps=500, render=False
        ),
    ]
    trainer = Trainer(callbacks, dataloader)
    trainer.train()
