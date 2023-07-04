import numpy as np

from gymnasium import Env, spaces
from gymnasium.utils import seeding


class ForgetMeNot(Env):
    def __init__(
        self,
        num_actions,
        game_length=2,
        min_memory_gap=1,
        max_memory_gap=5,
        hard_mode=False,
    ):
        """
        This game involves memorising a pattern for a set number of steps. On the last step the agent has
        to output an action that matches this pattern.
        * The pattern length should be <= to the number of actions that the network can output.
        * Set min_memory_gap and max_memory_gap to 0 to make a variant that doesn't require memory.
        * Increase min_memory_gap and max_memory_gap to make the game more difficult.
        * For this to train the rollout length must be >= min_memory_gap
        """
        self._num_actions = num_actions
        self._min_memory_gap = min_memory_gap
        self._max_memory_gap = max_memory_gap
        assert game_length >= max_memory_gap
        self._game_length = game_length
        self._final_step = game_length - 1
        self._hard_mode = hard_mode

        high = np.array([1.0] * (num_actions + 2), dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            -np.ones(self._num_actions), np.ones(self._num_actions), dtype=np.float32
        )
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._step = 0
        self._memory_gap = np.random.randint(
            self._min_memory_gap, self._max_memory_gap + 1
        )
        self._memorise_start_step = self._game_length - self._memory_gap - 1
        self._pattern = np.random.uniform(-1.0, 1.0, self._num_actions)
        return self._create_observation(), {"info": []}

    def get_pattern(self):
        return self._pattern

    def get_memory_gap(self):
        return self._memory_gap

    def step(self, action):
        self._step += 1
        if self._step >= self._final_step:
            done = True
            error = np.abs(action - self._pattern)
            reward = np.exp(-2.0 * np.mean(error))

        else:
            done = False
            if self._hard_mode:
                reward = 0.0
            else:
                error = np.abs(action - self._pattern)
                reward = np.exp(-2.0 * np.mean(error))

        obs = self._create_observation()
        return obs, reward, done, False, {}

    def _create_observation(self):
        recall_flag = 0.0
        pattern_flag = 0.0
        # default the pattern to noise
        # pattern = np.random.uniform(-1.0, 1.0, self._pattern.shape) # NOISE
        pattern = np.zeros(self._pattern.shape)

        # If we are in the memorising stage set the pattern to the actual pattern
        if self._step < self._memorise_start_step:
            pattern_flag = 1.0
            pattern = self._pattern

        # If we are 1 step from the end, then ask the agent for the pattern
        if self._step >= (self._final_step - 1):
            recall_flag = 1.0
            # For the identity case, i.e. when memory is not required, send the pattern.
            # The agent can then learn to just copy the observation.
            if self._memory_gap == 0:
                pattern_flag = 1.0
                pattern = self._pattern

        state = np.concatenate(
            [
                np.array([pattern_flag]),  # 1 = The pattern is in the observation.
                np.array([recall_flag]),  # 1 = Next action should be the pattern.
                pattern,  # The pattern to memorise, or noise if pattern_flag == 0
            ]
        )

        return np.float32(state)

    def render(self, mode="human"):
        pass
