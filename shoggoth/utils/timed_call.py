"""
Simple block-based timers using Welford's Online Algorithm to approximate mean and variance.

Usage:
```python

timer = TimedBlock()

for _ in range(10):
    with timer():
        sleep(1)

print(time.sleep())

# (1.000013, 1.3e-5)
"""

from abc import abstractmethod, ABC
from typing import Type, Tuple, Dict
from collections import deque, defaultdict
import time
from dataclasses import dataclass, field
import numpy as np


class StatisticsAccumulator(ABC):
    """Interface for a statistics integrator"""

    @abstractmethod
    def add(self, value: float):
        """Add the `value` to the running statistics

        :param value: the sample to integrate
        """
        ...

    @abstractmethod
    def current(self) -> Tuple[float, float]:
        """Returns the statistics of the observed samples so far

        :returns: a tuple (mean, variance)"""
        ...


@dataclass
class WelfordAccumulator(StatisticsAccumulator):
    """Implements Welford's Online Algorithm for single-pass variance and mean"""

    count: int = 0
    mean: float = 0.0
    differences: float = 0.0

    def add(self, value: float):
        """Add the `value` to the running statistics

        :param value: the sample to integrate
        """
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.differences += delta * delta2

    def current(self) -> Tuple[float, float]:
        """Returns the current values of the Welford algorithm

        :returns: a tuple (mean, variance)"""
        if self.count == 0:
            return float("nan"), float("nan")

        return self.mean, self.differences / self.count


@dataclass
class MovingWindowAccumulator(StatisticsAccumulator):
    values: deque = field(default_factory=lambda: deque(maxlen=100))

    def add(self, value: float):
        """Add the `value` to the running statistics

        :param value: the sample to integrate
        """
        self.values.append(value)

    def current(self) -> Tuple[float, float]:
        """Returns the current statistics

        :returns: a tuple (mean, variance)"""

        if len(self.values) == 0:
            return float("nan"), float("nan")

        return np.mean(self.values), np.var(self.values)


class TimedBlock:
    """Used to track the performance statistics of a block of code, in terms
    of execution time."""

    def __init__(
        self, tracker_type: Type[StatisticsAccumulator] = MovingWindowAccumulator
    ):
        """Create a new timed block instance

        :param tracker_type: The statistics integrator to use. Defaults to to MovingWindowStats"""
        self._tracker = tracker_type()
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()

    def __exit__(self, *args):
        self._tracker.add(time.perf_counter() - self._start)

    def mean(self) -> float:
        """Retrieve the mean execution time"""
        return self._tracker.current()[0]

    def var(self):
        """Retrieve the variance of the execution time"""
        return self._tracker.current()[1]

    def stats(self):
        """Retrieve the mean and the variance of execution time"""
        return self._tracker.current()


class BlockTimers:
    def __init__(
        self, tracker_type: Type[StatisticsAccumulator] = MovingWindowAccumulator
    ):
        self._timers: Dict[str, TimedBlock] = defaultdict(
            lambda: TimedBlock(tracker_type)
        )

    def scope(self, name: str) -> TimedBlock:
        return self._timers[name]

    def stats(self):
        return {name: timer.stats() for name, timer in self._timers.items()}


__all__ = ["TimedBlock", "MovingWindowAccumulator", "WelfordAccumulator"]
