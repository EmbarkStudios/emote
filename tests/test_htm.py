import torch

from gymnasium.vector import AsyncVectorEnv
from torch import nn
from torch.optim import Adam

from emote import Trainer
from emote.callbacks.logging import TerminalLogger
from emote.callbacks.testing import FinalRewardTestCheck
from emote.extra.onnx_exporter import OnnxExporter
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsTable
from emote.nn.gaussian_policy import GaussianMlpPolicy as Policy
from emote.sac import AlphaLoss, FeatureAgentProxy, PolicyLoss, QLoss, QTarget

from .gym import DictGymWrapper, HitTheMiddle, SimpleGymCollector


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


def test_htm():
    device = torch.device("cpu")
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(spaces=env.dict_space, maxlen=1000, device=device)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table, 100, 2, "batch_size")

    q1 = QNet(2, 1)
    q2 = QNet(2, 1)
    policy = Policy(2, 1, [N_HIDDEN, N_HIDDEN])
    ln_alpha = torch.tensor(1.0, requires_grad=True)
    agent_proxy = FeatureAgentProxy(policy, device)

    logged_cbs = [
        QLoss(name="q1", q=q1, opt=Adam(q1.parameters(), lr=8e-3)),
        QLoss(name="q2", q=q2, opt=Adam(q2.parameters(), lr=8e-3)),
        PolicyLoss(pi=policy, ln_alpha=ln_alpha, q=q1, opt=Adam(policy.parameters())),
        AlphaLoss(pi=policy, ln_alpha=ln_alpha, opt=Adam([ln_alpha]), n_actions=1),
        QTarget(pi=policy, ln_alpha=ln_alpha, q1=q1, q2=q2),
    ]

    callbacks = logged_cbs + [
        SimpleGymCollector(env, agent_proxy, memory_proxy, warmup_steps=500, render=False),
        TerminalLogger(logged_cbs, 400),
        FinalRewardTestCheck(logged_cbs[4], -5.0, 2000),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    env.close()


def test_htm_onnx(tmpdir):
    device = torch.device("cpu")
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(spaces=env.dict_space, maxlen=1000, device=device)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table, 100, 2, "batch_size")

    q1 = QNet(2, 1)
    q2 = QNet(2, 1)
    policy = Policy(2, 1, [N_HIDDEN, N_HIDDEN])
    ln_alpha = torch.tensor(1.0, requires_grad=True)
    agent_proxy = FeatureAgentProxy(policy, device)

    exporter = OnnxExporter(
        agent_proxy,
        env.dict_space,
        True,
        tmpdir / "inference",
        400,
    )

    logged_cbs = [
        QLoss(name="q1", q=q1, opt=Adam(q1.parameters(), lr=8e-3)),
        QLoss(name="q2", q=q2, opt=Adam(q2.parameters(), lr=8e-3)),
        PolicyLoss(pi=policy, ln_alpha=ln_alpha, q=q1, opt=Adam(policy.parameters())),
        AlphaLoss(pi=policy, ln_alpha=ln_alpha, opt=Adam([ln_alpha]), n_actions=1),
        QTarget(pi=policy, ln_alpha=ln_alpha, q1=q1, q2=q2),
    ]

    callbacks = logged_cbs + [
        exporter,
        SimpleGymCollector(env, agent_proxy, memory_proxy, warmup_steps=500, render=False),
        TerminalLogger(logged_cbs, 400),
        FinalRewardTestCheck(logged_cbs[4], -5.0, 2000),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    env.close()
