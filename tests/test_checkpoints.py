from os.path import join
from tempfile import mkdtemp
from typing import Generator

import torch

from torch import nn
from torch.optim import Adam

from emote import Trainer
from emote.callbacks import BackPropStepsTerminator, Checkpointer, CheckpointLoader
from emote.sac import QLoss
from emote.trainer import TrainingShutdownException


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


def nostep_dataloader() -> Generator:
    raise TrainingShutdownException()
    yield {}  # Needed to make this a generator.


def onestep_dataloader() -> Generator:
    yield {}
    raise TrainingShutdownException()


def test_networks_checkpoint():
    chkpt_dir = mkdtemp()
    run_root = join(chkpt_dir, "chkpt")
    n1 = nn.Linear(1, 1)
    c1 = [
        Checkpointer(
            networks=[n1], callbacks=[], run_root=run_root, checkpoint_interval=1
        )
    ]

    t1 = Trainer(c1, onestep_dataloader())
    t1.state["inf_step"] = 0
    t1.state["bp_step"] = 0
    t1.state["batch_size"] = 0
    t1.train()
    n2 = nn.Linear(1, 1)
    test_data = torch.rand(5, 1)
    assert not torch.allclose(n1(test_data), n2(test_data))

    c2 = [
        CheckpointLoader(
            networks=[n2], callbacks=[], run_root=run_root, checkpoint_index=0
        ),
        BackPropStepsTerminator(1),
    ]
    t2 = Trainer(c2, nostep_dataloader())
    t2.train()
    assert torch.allclose(n1(test_data), n2(test_data))


def random_onestep_dataloader() -> Generator:
    yield {
        "default": {
            "observation": {"obs": torch.rand(3, 2)},
            "actions": torch.rand(3, 1),
            "q_target": torch.ones(3, 1),
        },
    }
    raise TrainingShutdownException()


def test_qloss_checkpoints():
    chkpt_dir = mkdtemp()
    run_root = join(chkpt_dir, "chkpt")
    q1 = QNet(2, 1)
    ql1 = QLoss(name="q", q=q1, opt=Adam(q1.parameters()))
    c1 = [
        ql1,
        Checkpointer(
            networks=[], callbacks=[ql1], run_root=run_root, checkpoint_interval=1
        ),
    ]

    t1 = Trainer(c1, random_onestep_dataloader())
    t1.state["inf_step"] = 0
    t1.state["bp_step"] = 0
    t1.state["batch_size"] = 0
    t1.train()
    q2 = QNet(2, 1)
    test_obs = torch.rand(5, 2)
    test_act = torch.rand(5, 1)
    assert not torch.allclose(q1(test_act, test_obs), q2(test_act, test_obs))

    ql2 = QLoss(name="q", q=q2, opt=Adam(q1.parameters()))
    c2 = [
        ql2,
        CheckpointLoader(
            networks=[], callbacks=[ql2], run_root=run_root, checkpoint_index=0
        ),
    ]
    t2 = Trainer(c2, nostep_dataloader())
    t2.train()
    assert torch.allclose(q1(test_act, test_obs), q2(test_act, test_obs))
