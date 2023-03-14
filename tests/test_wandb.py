import os

from emote.callbacks import Callback, LoggingMixin, WBLogger
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


def test_logging():
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
    assert logger.wandb.run is not None

    # check if the additional info is logged in the config
    assert "metadata" in logger.wandb.config.keys()

    Trainer([dummy_cb, logger], DummyLoader()).train()

    # wandb.summary() is a dict that should contain the last logged values of the run by default
    assert logger.wandb.summary["dummy_bp_step"] == N
