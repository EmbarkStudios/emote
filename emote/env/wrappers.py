from collections import deque

import cv2
import gymnasium
import numpy as np

from gymnasium import spaces


cv2.ocl.setUseOpenCL(False)

# Note:
#
# Most of these wrappers are from the openai baselines repository:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
#
# Copyright (c) 2017 OpenAI (http://openai.com), used under the MIT license


class WarpFrame(gymnasium.ObservationWrapper):
    def __init__(self, env, width: int = 84, height: int = 84):
        """Warp frames to width x height.

        :param env: (Gym Environment) the environment
        """
        gymnasium.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame):
        """Returns the current observation from a frame.

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gymnasium.Wrapper):
    def __init__(self, env, n_frames: int):
        """Stack n_frames last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        LazyFrames (Below)

        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gymnasium.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * n_frames),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gymnasium.ObservationWrapper):
    def __init__(self, env):
        gymnasium.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return ((np.array(observation).astype(np.float32) / 255.0) - 0.5) * 2.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are
        only stored once. It exists purely to optimize memory usage which can
        be huge for DQN's 1M frames replay buffers.

        This object should only be converted to np.ndarray before being
        passed to the model.

        :param frames: ([int] or [float]) environment frames
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
