import math

from dataclasses import dataclass

from emote.utils.math import truncated_linear


@dataclass
class BPStepScheduler:
    bp_step_begin: float
    bp_step_end: float
    value_min: float
    value_max: float

    def evaluate_at(self, bp):
        return truncated_linear(
            self.bp_step_begin, self.bp_step_end, self.value_min, self.value_max, bp
        )


class Schedule:
    def __init__(self, initial: float, final: float, steps: int):
        self.initial = initial
        self.final = final
        self.steps = steps

        self._step_count = 0
        self._last_val = initial

    def get_last_val(self) -> float:
        return self._last_val

    def step(self):
        pass


class ConstantSchedule(Schedule):
    """Constant value that doesn't change over time.

    Args:
        value (float): Value of the schedule.
    """

    def __init__(
        self,
        value,
    ):
        super().__init__(value, None, None)


class LinearSchedule(Schedule):
    """Linear interpolation between initial and final over steps timesteps.
    After this many timesteps, final is returned.

    Args:
        initial (float): Initial value.
        final (float): Final value.
        steps (int): Number of steps.
        use_staircase (bool, optional): Use step like decay. Defaults to False.
        staircase_steps (int, optional): The number of discrete steps. Defaults to 5.
    """

    def __init__(
        self,
        initial: float,
        final: float,
        steps: int,
        use_staircase: bool = False,
        staircase_steps: int = 5,
    ):
        super().__init__(initial, final, steps)

        self.use_staircase = use_staircase
        self.staircase_steps = staircase_steps

    def step(self):
        fraction = self._step_count / self.steps
        if self.use_staircase:
            fraction = math.floor(fraction * self.staircase_steps) / self.staircase_steps
        fraction = min(fraction, 1.0)

        self._last_val = self.initial + fraction * (self.final - self.initial)

        self._step_count += 1


class CyclicSchedule(Schedule):
    """Cyclic schedule.

    Args:
        initial (float): Initial value.
        final (float): Final value.
        half_period_steps (int): Number of steps in one half of the cycle.
        mode (str, optional): One of {triangular, triangular2}. Defaults to "triangular".

        * triangular: A basic triangular cycle without amplitude scaling.
        * triangular2: A basic triangular cycle that scales initial amplitude by half each cycle.

        ** Note: for triangular2, the final value is the boundary that is scaled down
        at each cycle iteration,
        meaning that the value of the scheduled parameter will settle around initial.
    """

    def __init__(
        self,
        initial: float,
        final: float,
        half_period_steps: int,
        mode: str = "triangular",
    ):
        super().__init__(initial, final, half_period_steps)

        self.mode = mode

        if self.mode == "triangular":
            self.scale_fn = self._triangular_scale_fn
        elif self.mode == "triangular2":
            self.scale_fn = self._triangular2_scale_fn

    def _triangular_scale_fn(self, x: float) -> float:
        return 1

    def _triangular2_scale_fn(self, x: float) -> float:
        return 1 / (2.0 ** (x - 1))

    def step(self):
        cycle = math.floor(1 + self._step_count / (2 * self.steps))
        x = math.fabs(self._step_count / self.steps - 2 * cycle + 1)

        self._last_val = self.initial + (self.final - self.initial) * max(
            0, (1 - x)
        ) * self.scale_fn(cycle)

        self._step_count += 1


class CosineAnnealing(Schedule):
    """Cosine annealing schedule.

    Args:
        initial (float): Initial value.
        final (float): Final value.
        steps (int): Number of steps.
    """

    def __init__(self, initial: float, final: float, steps: int):
        super().__init__(initial, final, steps)

    def step(self):
        if self._step_count > 0:
            if (self._step_count - 1 - self.steps) % (2 * self.steps) == 0:
                self._last_val += (
                    (self.initial - self.final) * (1 - math.cos(math.pi / self.steps)) / 2
                )
            else:
                self._last_val = (1 + math.cos(math.pi * self._step_count / self.steps)) / (
                    1 + math.cos(math.pi * (self._step_count - 1) / self.steps)
                ) * (self._last_val - self.final) + self.final

        self._step_count += 1


class CosineAnnealingWarmRestarts(Schedule):
    """Cosine annealing schedule with warm restarts.

    Args:
        initial (float): Initial value.
        final (float): Final value.
        steps (int): Number of steps.
    """

    def __init__(self, initial: float, final: float, steps: int):
        super().__init__(initial, final, steps)

    def step(self):
        if self._step_count >= self.steps:
            self._step_count %= self.steps

        self._last_val = (
            self.final
            + (self.initial - self.final)
            * (1 + math.cos(math.pi * self._step_count / self.steps))
            / 2
        )

        self._step_count += 1
