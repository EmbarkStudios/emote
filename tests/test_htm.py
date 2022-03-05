import torch
from torch import nn
from torch.optim import Adam
from gym.vector import AsyncVectorEnv

from shoggoth import Trainer
from shoggoth.callbacks import (
    FinalLossTestCheck,
    TerminalLogger,
)
from shoggoth.nn import GaussianPolicyHead
from shoggoth.memory.builder import DictObsTable
from shoggoth.sac import (
    QLoss,
    QTarget,
    PolicyLoss,
    AlphaLoss,
    FeatureAgentProxy,
)
from shoggoth.memory import TableMemoryProxy, MemoryLoader

from .gym import SimpleGymCollector, HitTheMiddle, HiveGymWrapper


N_HIDDEN = 10


class QNet(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs + act, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, 1),
        )

    def forward(self, action, obs):
        x = torch.cat([obs, action], dim=1)
        return self.q(x)


class Policy(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.ReLU(),
            GaussianPolicyHead(N_HIDDEN, act),
        )

    def forward(self, obs):
        return self.pi(obs)


def test_htm():

    env = HiveGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(spaces=env.hive_space, maxlen=1000)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table, 100, 2, "batch_size")

    q1 = QNet(2, 1)
    q2 = QNet(2, 1)
    policy = Policy(2, 1)
    ln_alpha = torch.tensor(1.0, requires_grad=True)
    agent_proxy = FeatureAgentProxy(policy)

    logged_cbs = [
        QLoss(name="q1", q=q1, opt=Adam(q1.parameters(), lr=8e-3)),
        QLoss(name="q2", q=q2, opt=Adam(q2.parameters(), lr=8e-3)),
        PolicyLoss(pi=policy, ln_alpha=ln_alpha, q=q1, opt=Adam(policy.parameters())),
        AlphaLoss(pi=policy, ln_alpha=ln_alpha, opt=Adam([ln_alpha]), n_actions=1),
        QTarget(pi=policy, ln_alpha=ln_alpha, q1=q1, q2=q2),
    ]

    callbacks = logged_cbs + [
        SimpleGymCollector(env, agent_proxy, memory_proxy, warmup_steps=500),
        TerminalLogger(logged_cbs, 400),
        FinalLossTestCheck([logged_cbs[2]], [10.0], 2000),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()
