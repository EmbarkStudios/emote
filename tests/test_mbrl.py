import torch
from torch import nn
from torch.optim import Adam
from emote.models.model import DynamicModel
from emote.models.ensemble import EnsembleOfGaussian
from emote.models.callbacks import ModelLoss, LossProgressCheck
from gymnasium.vector import AsyncVectorEnv
from emote import Trainer
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsTable
from emote.sac import FeatureAgentProxy
from tests.gym import DictGymWrapper, HitTheMiddle, SimpleGymCollector


def term_func(
        states: torch.Tensor,
):
    return torch.zeros(states.shape[0])


class RandomPolicy(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        rand_actions = 2 * (torch.rand(batch_size, self.action_dim) - 0.5)
        return rand_actions, 0


def test_ensemble_trainer():
    device = torch.device("cpu")
    batch_size = 5
    rollout_length = 1
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    num_obs = 2
    num_actions = 1
    table = DictObsTable(
        spaces=env.dict_space,
        maxlen=10000,
        device=device)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table=table, rollout_count=batch_size // rollout_length,
                              rollout_length=rollout_length, size_key="batch_size")

    model = EnsembleOfGaussian(
        in_size=num_obs + num_actions,
        out_size=num_obs + 1,
        device=device,
        ensemble_size=5,
    )
    dynamic_model = DynamicModel(model=model)
    policy = RandomPolicy(action_dim=1)
    agent_proxy = FeatureAgentProxy(policy, device)

    callbacks = [
        ModelLoss(
            model=dynamic_model, opt=Adam(dynamic_model.model.parameters())
        ),
        LossProgressCheck(
            model=dynamic_model, num_bp=500
        ),
        SimpleGymCollector(
            env, agent_proxy, memory_proxy, warmup_steps=500, render=False
        ),
    ]
    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    env.close()


if __name__ == "__main__":
    test_ensemble_trainer()
