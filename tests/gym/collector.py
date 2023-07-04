"""
Collectors for running OpenAI gym environments
"""

import threading

from collections import deque

import numpy as np

from tests.gym.dict_gym_wrapper import DictGymWrapper

from emote.callback import Callback
from emote.mixins.logging import LoggingMixin
from emote.proxies import AgentProxy, MemoryProxy
from emote.typing import EpisodeState


class GymCollector(LoggingMixin, Callback):
    MAX_NUMBER_REWARDS = 1000

    def __init__(
        self,
        env: DictGymWrapper,
        agent: AgentProxy,
        memory: MemoryProxy,
        render: bool = True,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self._agent = agent
        self._memory = memory
        self._env = env
        self._render = render
        self._last_environment_rewards = deque(maxlen=1000)
        self.num_envs = env.num_envs
        self._warmup_steps = warmup_steps

    def collect_data(self):
        """Collect a single rollout"""
        if self._render:
            self._env.render()
        actions = self._agent(self._obs)
        next_obs, ep_info = self._env.dict_step(actions)
        for o in next_obs.values():
            if o.rewards["reward"] is None:
                continue

            self.log_scalar("instant/reward", o.rewards["reward"])
            self.log_windowed_scalar("instant/smooth_reward", o.rewards["reward"])

            if o.episode_state == EpisodeState.TERMINAL:
                self.log_scalar("final/reward", o.rewards["reward"])
                self.log_windowed_scalar("final/smooth_reward", o.rewards["reward"])

        self._memory.add(self._obs, actions)
        self._obs = next_obs

        if "reward" in ep_info:
            self.log_scalar("episode/reward", ep_info["reward"])
            self.log_windowed_scalar("episode/smooth_reward", ep_info["reward"])

    def collect_multiple(self, count: int):
        """Collect multiple rollouts

        :param count: Number of rollouts to collect
        """
        for _ in range(count):
            self.collect_data()

    def begin_training(self):
        "Make sure all envs work and collect warmup steps."
        # Runs through the init, step cycle once on main thread to make sure all envs work.
        self._obs = self._env.dict_reset()
        actions = self._agent(self._obs)
        _ = self._env.dict_step(actions)
        self._obs = self._env.dict_reset()

        # Collect trajectories for warmup steps before starting training
        iterations_required = self._warmup_steps
        self.collect_multiple(iterations_required)


class ThreadedGymCollector(GymCollector):
    def __init__(
        self,
        env: DictGymWrapper,
        agent: AgentProxy,
        memory: MemoryProxy,
        render: bool = True,
        warmup_steps: int = 0,
    ):
        super().__init__(env, agent, memory, render, warmup_steps)
        self._warmup_steps = warmup_steps
        self._stop = False
        self._thread = None

    def collect_forever(self):
        """Collect rollouts forever

        .. warning::

            This function means forever when it says forever. There is no
            signal, internal or external, that'll cause this loop to end. You
            probably want to implement a loop that calls `collect_data` or
            `collect_multiple` while checking exit conditions.

        """
        # FIXME[tsolberg]: Works OK when subprocs are not involved, might want
        # to signal this (somehow). Responsibility of parent to wrap somehow?

        while not self._stop:
            self.collect_data()

    def begin_training(self):
        super().begin_training()
        self._thread = threading.Thread(target=self.collect_forever)
        self._thread.start()

    def end_training(self):
        self._stop = True

        if self._thread is not None:
            self._thread.join()
            self._thread = None

            # For subprocvecenv et al that need to close connections to not crash
            if hasattr(self._env, "close"):
                self._env.close()


class SimpleGymCollector(GymCollector):
    def __init__(
        self,
        env: DictGymWrapper,
        agent: AgentProxy,
        memory: MemoryProxy,
        render: bool = True,
        warmup_steps: int = 0,
        bp_steps_per_inf: int = 10,
    ):
        super().__init__(env, agent, memory, render, warmup_steps)
        self._bp_steps_per_inf = bp_steps_per_inf

    def begin_training(self):
        super().begin_training()
        return {"inf_step": self._warmup_steps}

    def begin_batch(self, inf_step, bp_step):
        if bp_step % self._bp_steps_per_inf == 0:
            self.collect_data()
        return {"inf_step": inf_step + self.num_envs}


class SimpleRecurrentGymCollector(GymCollector):
    def __init__(
        self,
        env: DictGymWrapper,
        agent: AgentProxy,
        memory: MemoryProxy,
        recurrent_key: str,
        initial_recurrent_state: np.ndarray,
        render: bool = True,
        warmup_steps: int = 0,
        bp_steps_per_inf: int = 10,
    ):
        super().__init__(env, agent, memory, render, warmup_steps)
        self._bp_steps_per_inf = bp_steps_per_inf
        self._recurrent_key = recurrent_key
        self._initial_recurrent_state = initial_recurrent_state

    def collect_data(self):
        """Collect a single rollout"""
        if self._render:
            self._env.render()

        actions = self._agent(self._obs)

        next_obs, ep_info = self._env.dict_step(actions)
        for o in next_obs.values():
            if o.rewards["reward"] is None:
                continue

            self.log_scalar("instant/reward", o.rewards["reward"])
            self.log_windowed_scalar("instant/smooth_reward", o.rewards["reward"])

            if o.episode_state == EpisodeState.TERMINAL:
                self.log_scalar("final/reward", o.rewards["reward"])
                self.log_windowed_scalar("final/smooth_reward", o.rewards["reward"])

        gru_states = {k: actions[k].list_data.pop(self._recurrent_key) for k in actions}
        self._memory.add(self._obs, actions)
        self._obs = next_obs
        for k, obs in self._obs.items():
            if k in actions:
                obs.array_data[self._recurrent_key] = gru_states[k]
            else:
                obs.array_data[self._recurrent_key] = self._initial_recurrent_state

        if "reward" in ep_info:
            self.log_scalar("episode/reward", ep_info["reward"])
            self.log_windowed_scalar("episode/smooth_reward", ep_info["reward"])

    def begin_training(self):
        "Make sure all envs work and collect warmup steps."
        # Runs through the init, step cycle once on main thread to make sure all envs work.
        self._obs = self._env.dict_reset()
        for obs in self._obs.values():
            obs.array_data[self._recurrent_key] = self._initial_recurrent_state

        actions = self._agent(self._obs)
        _ = self._env.dict_step(actions)
        self._obs = self._env.dict_reset()
        for obs in self._obs.values():
            obs.array_data[self._recurrent_key] = self._initial_recurrent_state

        # Collect trajectories for warmup steps before starting training
        iterations_required = self._warmup_steps
        self.collect_multiple(iterations_required)

        return {"inf_step": self._warmup_steps}

    def begin_batch(self, inf_step, bp_step):
        if bp_step % self._bp_steps_per_inf == 0:
            self.collect_data()
        return {"inf_step": inf_step + self.num_envs}
