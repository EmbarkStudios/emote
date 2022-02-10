import torch
from torch.optim import Adam
from gym.vector import SyncVectorEnv
from gym.wrappers import TransformObservation
from gym import spaces

from shoggoth import Trainer
from shoggoth.nn import ActionValue, GaussianMLPPolicy
from shoggoth.sac import (
    QLoss,
    QTarget,
    PolicyLoss,
    AlphaLoss,
    SACNetwork,
    FeatureAgentProxy,
)

from .gym import SimpleGymCollector, HitTheMiddle, ReplayMemory


def make_htm():
    env = TransformObservation(HitTheMiddle(), lambda obs: {"obs": obs})
    env.observation_space = spaces.Dict({"obs": env.env.observation_space})
    return env


def test_htm():

    env = SyncVectorEnv([make_htm, make_htm, make_htm])
    batch_size = 1000
    memory = ReplayMemory(10000, batch_size)

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
        SimpleGymCollector(env, agent_proxy, memory, warmup_steps=batch_size),
    ]

    trainer = Trainer(callbacks, memory, 200)
    trainer.train()
