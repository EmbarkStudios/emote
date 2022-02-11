"""
Collectors for running OpenAI gym environments
"""

import threading
from collections import deque
from gym.vector import SyncVectorEnv
from gym import spaces

from shoggoth.callback import Callback
from shoggoth.proxies import AgentProxy, Transitions, MemoryProxy


class GymCollector(Callback):
    MAX_NUMBER_REWARDS = 1000

    def __init__(
        self,
        env: SyncVectorEnv,
        agent: AgentProxy,
        memory: MemoryProxy,
        render: bool = True,
    ):
        super().__init__()
        self._agent = agent
        self._memory = memory
        self._env = env
        assert isinstance(env.observation_space, spaces.Dict), (
            "Observation spaces in shoggoth _must_ be of Dict type,\n"
            f"but the current env has observations of type\n{env.observation_space}."
        )
        self.num_envs = env.num_envs
        self._render = render
        self._last_environment_rewards = deque(maxlen=1000)
        self._rollouts = 0
        self._id_offset = {i: 0 for i in range(env.num_envs)}

    def _step(self):
        actions = self._agent(self._obs)
        next_obs, rewards, dones, _ = self._env.step(actions)
        agents = [env_id + self._id_offset[env_id] for env_id in range(self.num_envs)]
        self._memory.add(Transitions(self._obs, actions, rewards, agents, dones))

        if self._render:
            self._env.envs[0].render()

        for env_id, done in enumerate(dones):
            if done:
                self._id_offset[env_id] += self.num_envs

        self._obs = next_obs

    def collect_data(self):
        """Collect a single rollout"""
        self._step()

    def collect_multiple(self, count: int):
        """Collect multiple rollouts

        :param count: Number of rollouts to collect
        """
        for _ in range(count):
            self.collect_data()

    def begin_training(self):
        "Runs through the init, step cycle once on main thread to make sure all envs work."
        self._obs = self._env.reset()
        actions = self._agent(self._obs)
        _ = self._env.step(actions)
        self._obs = self._env.reset()


class ThreadedGymCollector(GymCollector):
    def __init__(self, env, agent, render=True):
        super().__init__(env, agent, render)
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
        env: SyncVectorEnv,
        agent: AgentProxy,
        memory: TransitionMemoryProxy,
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

    def begin_batch(self, inf_step, bp_step):
        if bp_step % self._bp_steps_per_inf == 0:
            self.collect_data()
        return {"inf_step": inf_step + self.num_envs}
