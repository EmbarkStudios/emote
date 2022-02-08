import torch
from torch.optim import Adam
from gym.vector import SyncVectorEnv

from shoggoth import Trainer
from shoggoth.nn import ActionValue, GaussianMLPPolicy
from shoggoth.sac import (
    QLoss,
    QTarget,
    PolicyLoss,
    AlphaLoss,
    SACNetwork,
    SACAgentProxy,
)

from .gym import SimpleGymCollector, HitTheMiddle, ReplayMemory


def test_htm():

    env = SyncVectorEnv([HitTheMiddle, HitTheMiddle])
    memory = ReplayMemory(10000, 1000)

    network = SACNetwork(
        ActionValue(2, 1, [10, 10]),
        ActionValue(2, 1, [10, 10]),
        ActionValue(2, 1, [10, 10]),
        ActionValue(2, 1, [10, 10]),
        GaussianMLPPolicy(2, 1, [10, 10]),
        torch.tensor(1.0),
    )
    agent_proxy = SACAgentProxy(network, memory)

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
        SimpleGymCollector(env, agent_proxy),
    ]

    trainer = Trainer(callbacks, memory, 200)
    trainer.train()
