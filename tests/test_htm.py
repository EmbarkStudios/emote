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
    policy = GaussianMLPPolicy(2, 1, [10, 10])
    ln_alpha = torch.tensor(1.0, requires_grad=True)
    agent_proxy = FeatureAgentProxy(policy)

    logged_cbs = [
        QLoss(name="q1", q=q1, opt=Adam(q1.parameters())),
        QLoss(name="q2", q=q2, opt=Adam(q2.parameters())),
        PolicyLoss(pi=policy, ln_alpha=ln_alpha, q=q1, opt=Adam(policy.parameters())),
        AlphaLoss(pi=policy, ln_alpha=ln_alpha, opt=Adam([ln_alpha]), n_actions=1),
        QTarget(pi=policy, ln_alpha=ln_alpha, q1=q1, q2=q2),
    ]

    callbacks = logged_cbs + [
        SimpleGymCollector(env, agent_proxy, memory_proxy, warmup_steps=1000),
        TerminalLogger(logged_cbs, 500),
    ]

    trainer = Trainer(callbacks, dataloader, 200)
    trainer.train()
