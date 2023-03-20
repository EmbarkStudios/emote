import torch
from torch import nn
from torch.optim import Adam
from emote.models.model import DynamicModel, ModelLoss
from emote.models.ensemble import EnsembleOfGaussian
from gymnasium.vector import AsyncVectorEnv
from emote import Trainer
from emote.callbacks import TerminalLogger
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsTable
from emote.sac import FeatureAgentProxy
from .gym import DictGymWrapper, HitTheMiddle, SimpleGymCollector


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
        return 2 * (torch.rand(batch_size, self.action_dim) - 0.5)


def test_model_learning():
    device = torch.device("cpu")
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    num_obs = 2
    num_actions = 1
    table = DictObsTable(spaces=env.dict_space, maxlen=1000, device=device)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table, 100, 2, "batch_size")

    model = EnsembleOfGaussian(
        in_size=num_obs + num_actions,
        out_size=num_obs + 1,
        device=device,
        ensemble_size=5,
    )
    dynamic_model = DynamicModel(model=model)
    policy = RandomPolicy(action_dim=1)
    agent_proxy = FeatureAgentProxy(policy, device)

    logged_cbs = [
        ModelLoss(model=dynamic_model, opt=Adam(dynamic_model.model.parameters()))
    ]

    callbacks = logged_cbs + [
        SimpleGymCollector(
            env, agent_proxy, memory_proxy, warmup_steps=500, render=False
        ),
        TerminalLogger(logged_cbs, 400),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    env.close()
