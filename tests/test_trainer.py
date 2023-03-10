from __future__ import annotations

import pytest

from emote.callback import Callback
from emote.trainer import Trainer, TrainingShutdownException


class DummyCallback(Callback):
    def __init__(self, cycle: int | None = None):
        super().__init__(cycle)
        self.end_cycle_called = 0

    def end_cycle(self):
        self.end_cycle_called += 1


class DummyLoader:
    def __iter__(self):
        for _ in range(3):
            yield {"batch_size": 0}

        raise TrainingShutdownException("end of data")


@pytest.mark.parametrize("interval,expected", ((None, 0), (0, 0), (1, 3), (2, 1)))
def test_callback_cycle_called_count(interval, expected):
    callback = DummyCallback(interval)

    Trainer([callback], DummyLoader()).train()

    assert callback.end_cycle_called == expected
