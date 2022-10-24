import random

import numpy as np

from gym import Env, spaces
from gym.utils import seeding


class HitTheMiddle(Env):
    def __init__(self):
        high = np.array([10, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        ones = np.ones(1, dtype=np.float32)
        self.action_space = spaces.Box(-ones, ones)
        self._state = None
        self._step = None
        self.viewer = None

    def step(self, action):
        self._state[1] += action
        self._state[0] += self._state[1]
        self._step += 1
        if self._state[0] > 10.0:
            self._state[0] = 10.0
            self._state[1] *= -1
        elif self._state[0] < -10.0:
            self._state[0] = -10.0
            self._state[1] *= -1
        done = False
        if self._step > 30:
            self._step = 0
            done = True

        return (
            self._state,
            float(-self._state[0] ** 2),
            done,
            {},
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        pos = random.random() * 20 - 10
        vel = random.random() * 0.5 - 0.25
        self._state = np.array([pos, vel])
        self._step = 0
        return self._state

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = 20
        scale = screen_width / world_width
        bally = screen_height / 2
        ballwidth = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            ball = rendering.make_circle(ballwidth / 2)
            self.balltrans = rendering.Transform()
            ball.add_attr(self.balltrans)
            ball.set_color(0.8, 0.1, 0.6)
            self.viewer.add_geom(ball)
            self.track = rendering.Line((0, bally), (screen_width, bally))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self._state is None:
            return None

        x = self._state[0]
        ballx = x * scale + screen_width / 2.0  # MIDDLE OF BALL
        self.balltrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))
