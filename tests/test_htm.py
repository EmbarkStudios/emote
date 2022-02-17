import numpy as np
import torch
from torch.optim import Adam
from gym.vector import SyncVectorEnv

from shoggoth import Trainer
from shoggoth.nn import ActionValue, GaussianMLPPolicy
from shoggoth.memory.builder import MemoryConfiguration, create_memory
from shoggoth.sac import (
    QLoss,
    QTarget,
    PolicyLoss,
    AlphaLoss,
    SACNetwork,
    FeatureAgentProxy,
)
from shoggoth.memory import TableMemoryProxy, MemoryLoader

from .gym import SimpleGymCollector, HitTheMiddle, HiveGymWrapper


def test_htm():

    env = HiveGymWrapper(SyncVectorEnv(3 * [HitTheMiddle]))
    batch_size = 1000
    mem_conf = MemoryConfiguration(10, 1000)
    table = create_memory(env.hive_space, mem_conf)
    sb = TableMemoryProxy(table)
    memory = MemoryLoader(table, 20, 2, 500, "batch_size")

    network = SACNetwork(
        ActionValue(2, 1, [10, 10]),
        ActionValue(2, 1, [10, 10]),
        ActionValue(2, 1, [10, 10]),
        ActionValue(2, 1, [10, 10]),
        GaussianMLPPolicy(2, 1, [10, 10]),
        torch.tensor(1.0),
    )
    agent_proxy = FeatureAgentProxy(network)

    callbacks = [
        QLoss(
            "q1",
            Adam(network.q1.parameters()),
            network.q1,
        ),
        QLoss(
            "q2",
            Adam(network.q2.parameters()),
            network.q2,
        ),
        PolicyLoss(
            "policy",
            Adam(network.policy.parameters()),
            network,
        ),
        AlphaLoss("alpha", Adam([network.log_alpha_vars]), network, 1),
        QTarget(
            network,
            0.99,
            1.0,
            0.005,
        ),
        SimpleGymCollector(env, agent_proxy, sb, warmup_steps=batch_size),
    ]

    trainer = Trainer(callbacks, memory, 200)
    trainer.train()
