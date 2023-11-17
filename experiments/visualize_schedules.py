import torch
import matplotlib.pyplot as plt
import argparse
from emote.extra.schedules import (
    CosineAnnealing,
    CosineAnnealingWarmRestarts,
    CyclicSchedule,
    LinearSchedule,
    Schedule,
)


class AdditiveSchedule(Schedule):
    def __init__(self, schedule1: Schedule, schedule2: Schedule = None):
        self._schedule1 = schedule1
        self._schedule2 = schedule2

    @property
    def value(self):
        if self._schedule2 is not None:
            return (self._schedule1.value + self._schedule2.value) / 2.0
        return self._schedule1.value

    def step(self):
        self._schedule1.step()
        if self._schedule2 is not None:
            self._schedule2.step()


def get_schedule(
    schedule_type: str,
    start: float,
    end: float,
    steps: int,
) -> Schedule:
    """Creates a schedule using the given values.

    Args:
        schedule_type (str): One of ("linear", "triangular", "triangular2",
            "cosine_annealing", "cosine_annealing_wr")
        start (float): Start value for the schedule.
        end (float): End value for the schedule.
        steps (int): Number of steps to get from start to end value.

    Returns:
        Schedule
    """
    if schedule_type == "linear":
        return LinearSchedule(initial=start, final=end, steps=steps)
    elif schedule_type == "triangular":
        return CyclicSchedule(
            initial=start,
            final=end,
            half_period_steps=steps,
            mode="triangular",
        )
    elif schedule_type == "triangular2":
        return CyclicSchedule(
            initial=start,
            final=end,
            half_period_steps=steps,
            mode="triangular2",
        )
    elif schedule_type == "cosine_annealing":
        return CosineAnnealing(initial=start, final=end, steps=steps)
    elif schedule_type == "cosine_annealing_wr":
        return CosineAnnealingWarmRestarts(initial=start, final=end, steps=steps)
    else:
        raise NotImplementedError(schedule_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--linear-start", type=float, default=-10)
    parser.add_argument("--linear-end", type=float, default=-20)
    parser.add_argument("--linear-steps", type=int, default=100_000)
    parser.add_argument("--schedule-type", type=str, default="cosine_annealing_wr")
    parser.add_argument("--schedule-start", type=float, default=-10)
    parser.add_argument("--schedule-end", type=float, default=-20)
    parser.add_argument("--schedule-steps", type=int, default=100_000)

    arg = parser.parse_args()

    steps = arg.steps

    linear_entropy_schedule = get_schedule(
        schedule_type="linear",
        start=arg.linear_start,
        end=arg.linear_end,
        steps=arg.linear_steps,
    )
    addition_entropy_schedule = get_schedule(
        schedule_type=arg.schedule_type,
        start=arg.schedule_start,
        end=arg.schedule_end,
        steps=arg.schedule_steps,
    )
    schedule = AdditiveSchedule(
        schedule1=linear_entropy_schedule, schedule2=addition_entropy_schedule
    )

    values = torch.zeros(steps)
    for i in range(steps):
        values[i] = schedule.value
        schedule.step()

    plt.plot(torch.arange(steps), values)
    plt.show()