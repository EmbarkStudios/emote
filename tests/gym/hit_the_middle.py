import random
import numpy as np
from gym import spaces
from gym import Env
from gym.utils import seeding


class HitTheMiddle(Env):
    def __init__(self):
        high = np.array([10, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        ones = np.ones(1, dtype=np.float32)
        self.action_space = spaces.Box(-ones, ones)
        self._pos = None
        self._vel = None
        self._step = None
        self.viewer = None

    def step(self, action):
        self._vel += action
        self._pos += self._vel
        self._step += 1
        if self._pos > 10.0:
            self._pos = 10.0
            self._vel *= -1
        elif self._pos < -10.0:
            self.pos = -10.0
            self._vel *= -1
        done = False
        if self._step > 30:
            self._step = 0
            done = True

        return (
            np.array((self._pos, self._vel), dtype=np.float32).flatten(),
            float(-self._pos**2),
            done,
            {},
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._pos = random.random() * 20 - 10
        self._vel = random.random() * 0.5 - 0.25
        self._step = 0
        return np.array((self._pos, self._vel), dtype=np.float32).flatten()

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

        if self._pos is None:
            return None

        x = self._pos
        ballx = x * scale + screen_width / 2.0  # MIDDLE OF BALL
        self.balltrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))
