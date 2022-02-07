from torch.nn import ParameterList
from torch.optim import Adam

from shoggoth import Trainer
from shoggoth.nn import ActionValue, GaussianMLPPolicy
from shoggoth.sac import QLoss, QTarget, PolicyLoss, AlphaLoss, SACNetwork


def test_htm():

    network = SACNetwork(
        ActionValue(2, 1, [10, 10]),
        ActionValue(2, 1, [10, 10]),
        ActionValue(2, 1, [10, 10]),
        ActionValue(2, 1, [10, 10]),
        GaussianMLPPolicy(2, 1, [10, 10]),
        ParameterList([1.0]),
    )

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
        AlphaLoss("alpha", Adam(network.log_alpha_vars), network, 1),
        QTarget(
            network,
            0.99,
            1.0,
            0.005,
        ),
    ]

    trainer = Trainer(callbacks)
    trainer.train()
