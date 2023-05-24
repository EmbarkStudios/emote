from typing import List

from emote.callback import Callback
from emote.callbacks.loss import LossCallback
from emote.mixins.logging import LoggingMixin
from emote.trainer import TrainingShutdownException


class FinalLossTestCheck(Callback):
    """Logs the provided loggable callbacks to the python logger."""

    def __init__(
        self,
        callbacks: List[LossCallback],
        cutoffs: List[float],
        test_length: int,
    ):
        super().__init__(cycle=test_length)
        self._cbs = callbacks
        self._cutoffs = cutoffs

    def end_cycle(self):
        for cb, cutoff in zip(self._cbs, self._cutoffs):
            loss = cb.scalar_logs[f"loss/{cb.name}_loss"]
            if loss > cutoff:
                raise Exception(f"Loss for {cb.name} too high: {loss}")
        raise TrainingShutdownException()


class FinalRewardTestCheck(Callback):
    def __init__(
        self,
        callback: LoggingMixin,
        cutoff: float,
        test_length: int,
    ):
        super().__init__(cycle=test_length)
        self._cb = callback
        self._cutoff = cutoff

    def end_cycle(self):
        reward = self._cb.scalar_logs["training/scaled_reward"]
        if reward < self._cutoff:
            raise Exception(f"Reward too low: {reward}")
        raise TrainingShutdownException()
