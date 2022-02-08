"""
Collectors for running OpenAI gym environments
"""

from collections import deque, defaultdict
import threading

from gym.vector import VectorEnv
import numpy as np

from shoggoth.callback import Callback
from shoggoth.proxies import AgentProxy, Observations, Responses


class GymCollector(Callback):
    MAX_NUMBER_REWARDS = 1000

    def __init__(self, env: VectorEnv, agent_proxy: AgentProxy, render: bool = True):
        super().__init__()
        self._agent_proxy = agent_proxy
        self._env = env
        self._n_steps = env.num_envs
        self._render = render
        self._last_environment_rewards = deque(maxlen=1000)
        self._rollouts = 0
        self._generations = {i: 0 for i in range(env.num_envs)}

    def _reset_obs(self, obs_space, observations):
        obs = np.zeros(
            (self._env.num_envs,) + obs_space.shape, dtype=obs_space.dtype.name
        )
        obs[:] = observations
        return {"obs": obs}

    def gym_to_shoggoth(self, data: np.array) -> Observations:
        print(data)
        return {
            f"{env + self._generations[env] * self._n_steps}": data[env]
            for env in range(self._env.num_envs)
        }

    def shoggoth_to_gym(self, data: Responses):
        return np.array([action["action"] for _, action in data.items()])

    def _step(self, ep_infos):
        actions = self._agent_proxy(self.gym_to_shoggoth(self._obs))
        next_obs, rewards, dones, infos = self._env.step(actions)

        if self._render:
            self._env.envs[0].render()

        for env_id, done in enumerate(dones):
            if done:
                self._generations[env_id] += 1

        self._obs = {"obs": next_obs}

    def collect_data(self):
        """Collect a single rollout"""
        ep_infos = {"agent_metrics": []}

        inference_steps = 0
        completed_episodes = 0
        for _ in range(self._n_steps):
            new_inference_steps, new_completed_episodes = self._step(ep_infos)
            inference_steps += new_inference_steps
            completed_episodes += new_completed_episodes

    def collect_multiple(self, count: int):
        """Collect multiple rollouts

        :param count: Number of rollouts to collect
        """
        for _ in range(count):
            self.collect_data()

    def begin_training(self):
        "Runs through the init, step cycle once on main thread to make sure all envs work."
        observations = self._env.reset()
        self._obs = self._reset_obs(self._env.observation_space, observations)
        actions = self._agent_proxy(self.gym_to_shoggoth(self._obs))
        _ = self._env.step(actions)

        observations = self._env.reset()
        self._obs = self._reset_obs(self._env.observation_space, observations)


class ThreadedGymCollector(GymCollector):
    def __init__(self, env, agent_proxy, render=True):
        super().__init__(env, agent_proxy, render)
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
        env: VectorEnv,
        agent_proxy,
        render=True,
        bp_steps_per_inf=10,
        warmup_steps=0,
    ):
        super().__init__(env, agent_proxy, render)
        self._warmup_steps = warmup_steps
        self._bp_steps_per_inf = bp_steps_per_inf

    def begin_training(self):
        super().begin_training()
        iterations_required = self._warmup_steps // self._n_steps
        self.collect_multiple(iterations_required)
        return {"inf_step": self._warmup_steps}

    def begin_batch(self, inf_step, bp_step):
        if bp_step % self._bp_steps_per_inf == 0:
            self.collect_data()
        return {"inf_step": inf_step + self._n_steps}
