import gym

from emote.env.wrappers import (
    FrameStack,
    ScaledFloatFrame,
    WarpFrame,
)


def make_vision_box2d_env(
    environment_id: str,
    rank: int,
    seed: int = 0,
    frame_stack: int = 3,
    use_float_scaling: bool = True,
):
    """
    :param environment_id: (str) the environment ID
    :param rank: (int) an integer offset for the random seed
    :param seed: (int) the inital seed for RNG
    :param frame_stack: (int) Stacks this many frames.
    :param use_float_scaling: (bool) scaled the observations from char to normalised float
    :return: the env creator function
    """

    def _thunk():
        env = gym.make(environment_id)
        env.seed(seed + rank)
        env = WarpFrame(env)
        if use_float_scaling:
            env = ScaledFloatFrame(env)

        if frame_stack > 1:
            env = FrameStack(env, frame_stack)
        return env

    return _thunk
