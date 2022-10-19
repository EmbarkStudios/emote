from typing import Dict

import gym

from emote.env.wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    FrameStack,
    MaxAndSkipEnv,
    NoopResetEnv,
    ScaledFloatFrame,
    TupeliserFrame,
    WarpFrame,
)


def make_vision_atari_env(
    environment_id: str,
    rank: int,
    seed: int = 2022,
    make_tuple: bool = False,
    episode_life: bool = True,
    clip_rewards: bool = True,
    frame_stack: int = 3,
    scale: bool = True,
):
    """
    Create an atari env

    :param environment_id: (str) the environment ID
    :param rank: (int) an integer offset for the random seed
    :param seed: (int) the inital seed for RNG
    :param make_tuple: (bool) Makes the observation a Tuple.
    :param episode_life: (bool) Makes death terminal.
    :param clip_rewards: (bool) Clips the environment rewards to {+1, 0, -1} by its sign.
    :param frame_stack: (int) Stacks this many frames.
    :return: the env creator function
    """
    assert frame_stack > 1, "Frame stack must be greater than 1"

    def _thunk():
        env = gym.make(environment_id)
        assert "NoFrameskip" in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env.seed(seed + rank)
        if episode_life:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        if scale:
            env = ScaledFloatFrame(env)
        if clip_rewards:
            env = ClipRewardEnv(env)
        if frame_stack > 1:
            env = FrameStack(env, frame_stack)
        if make_tuple:
            return TupeliserFrame(env)
        else:
            return env

    return _thunk
