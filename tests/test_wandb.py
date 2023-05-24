import os

import pytest
import wandb

from emote.callback import Callback
from emote.mixins.logging import LoggingMixin
from emote.trainer import Trainer


os.environ["WANDB_MODE"] = "offline"
N = 10000


class DummyCallback(LoggingMixin, Callback):
    def __init__(self):
        super().__init__()
        self.end_batch_called = 0

    def end_batch(self):
        self.end_batch_called += 1
        self.log_scalar("dummy", self.end_batch_called)


class DummyLoader:
    def __iter__(self):
        for _ in range(N):
            yield {"batch_size": 0}


def test_raises_help_if_wandb_not_installed(hide_pkg):
    hide_pkg("wandb")

    with pytest.raises(ImportError) as ex:
        from emote.callbacks.wb_logger import WBLogger

        WBLogger([], {}, log_interval=1)

    assert ex.value.msg == "enable the optional `wandb` feature to use the WBLogger"
    assert isinstance(ex.value.__cause__, ImportError)
    assert ex.value.__cause__.msg == "No module named 'wandb'"


def test_logging():
    from emote.callbacks.wb_logger import WBLogger

    dummy_cb = DummyCallback()
    logger = WBLogger(
        callbacks=[dummy_cb],
        config={
            "wandb_project": "test_project",
            "wandb_run": "test_run",
            "metadata": "test",
        },
        log_interval=N,
    )

    # check if a run is initialized
    assert wandb.run is not None

    # check if the additional info is logged in the config
    assert "metadata" in wandb.config.keys()

    Trainer([dummy_cb, logger], DummyLoader()).train()

    # wandb.summary() is a dict that should contain the last logged values of the run by default
    assert wandb.summary["dummy_bp_step"] == N
