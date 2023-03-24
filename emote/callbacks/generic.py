from emote.callback import Callback
from emote.trainer import TrainingShutdownException


class BackPropStepsTerminator(Callback):
    """Terminates training after a given number of backprops.

    :param bp_steps (int): The total number of backprops that the trainer should run
        for.
    """

    def __init__(self, bp_steps: int):
        assert bp_steps > 0, "Training steps must be above 0."
        super().__init__(cycle=bp_steps)

    def end_cycle(self):
        raise TrainingShutdownException()
