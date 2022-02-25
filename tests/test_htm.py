import torch
from torch.optim import Adam
from gym.vector import SyncVectorEnv

from shoggoth import Trainer
from shoggoth.callbacks import TerminalLogger
from shoggoth.nn import ActionValue, GaussianMLPPolicy
from shoggoth.memory.builder import MemoryConfiguration, create_memory
from shoggoth.sac import (
    QLoss,
    QTarget,
    PolicyLoss,
    AlphaLoss,
    FeatureAgentProxy,
)
from shoggoth.memory import TableMemoryProxy, MemoryLoader

from .gym import SimpleGymCollector, HitTheMiddle, HiveGymWrapper


def test_htm():

    env = HiveGymWrapper(SyncVectorEnv(3 * [HitTheMiddle]))
    mem_conf = MemoryConfiguration(10, 1000)
    table = create_memory(env.hive_space, mem_conf)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table, 20, 2, 500, "batch_size")

    q1 = ActionValue(2, 1, [10, 10])
    q2 = ActionValue(2, 1, [10, 10])
    q1t = ActionValue(2, 1, [10, 10])
    q2t = ActionValue(2, 1, [10, 10])
    policy = GaussianMLPPolicy(2, 1, [10, 10])
    ln_alpha = torch.tensor(1.0, requires_grad=True)
    agent_proxy = FeatureAgentProxy(policy)

    callbacks = [
        QLoss("q1", q1, Adam(q1.parameters())),
        QLoss("q2", q2, Adam(q2.parameters())),
        PolicyLoss(policy, ln_alpha, q1, q2, Adam(policy.parameters())),
        AlphaLoss(policy, ln_alpha, Adam([ln_alpha]), 1),
        QTarget(policy, q1t, q2t, ln_alpha, q1, q2),
    ]

    callbacks += [
        SimpleGymCollector(env, agent_proxy, memory_proxy, warmup_steps=1000),
        TerminalLogger(callbacks, 500),
    ]

    trainer = Trainer(callbacks, dataloader, 200)
    trainer.train()
