"""
Collectors for running OpenAI gym environments
"""

import threading
from collections import deque

from emote.callback import Callback
from emote.proxies import AgentProxy, MemoryProxy
from tests.gym.dict_gym_wrapper import DictGymWrapper


class GymCollector(Callback):
    MAX_NUMBER_REWARDS = 1000

    def __init__(
        self,
        env: DictGymWrapper,
        agent: AgentProxy,
        memory: MemoryProxy,
        render: bool = True,
    ):
        super().__init__()
        self._agent = agent
        self._memory = memory
        self._env = env
        self._render = render
        self._last_environment_rewards = deque(maxlen=1000)
        self.num_envs = env.num_envs

    def collect_data(self):
        """Collect a single rollout"""
        if self._render:
            self._env.render()
        actions = self._agent(self._obs)
        next_obs = self._env.dict_step(actions)
        self._memory.add(self._obs, actions)
        self._obs = next_obs

    def collect_multiple(self, count: int):
        """Collect multiple rollouts

        :param count: Number of rollouts to collect
        """
        for _ in range(count):
            self.collect_data()

    def begin_training(self):
        "Runs through the init, step cycle once on main thread to make sure all envs work."
        self._obs = self._env.dict_reset()
        actions = self._agent(self._obs)
        _ = self._env.dict_step(actions)
        self._obs = self._env.dict_reset()


class ThreadedGymCollector(GymCollector):
    def __init__(
        self,
        env: DictGymWrapper,
        agent: AgentProxy,
        memory: MemoryProxy,
        render=True,
        warmup_steps=0,
    ):
        super().__init__(env, agent, memory, render)
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
        # Collect trajectories for warmup steps before starting training
        super().begin_training()
        iterations_required = self._warmup_steps
        self.collect_multiple(iterations_required)

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
        render=True,
        bp_steps_per_inf=10,
        warmup_steps=0,
    ):
        super().__init__(env, agent, memory, render)
        self._warmup_steps = warmup_steps
        self._bp_steps_per_inf = bp_steps_per_inf

    def begin_training(self):
        super().begin_training()
        iterations_required = self._warmup_steps
        self.collect_multiple(iterations_required)
        return {"inf_step": self._warmup_steps}

    def begin_batch(self, inf_step, bp_steps):
        if bp_steps % self._bp_steps_per_inf == 0:
            self.collect_data()
        return {"inf_step": inf_step + self.num_envs}
