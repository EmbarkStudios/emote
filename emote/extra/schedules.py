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
