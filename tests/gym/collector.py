"""
Collectors for running OpenAI gym environments
"""

from collections import deque, defaultdict
import threading

import gym
import numpy as np

from shoggoth.callback import Callback


class GymCollector(Callback):
    MAX_NUMBER_REWARDS = 1000

    def __init__(self, env: gym.vector.VectorEnv, agent_proxy, render=True):
        self._agent_proxy = agent_proxy
        self._env = env
        self._n_steps = env.num_envs
        self._render = render
        self._last_environment_rewards = deque(maxlen=1000)
        self._rollouts = 0
        self._generations = defaultdict(int)

    def _reset_obs(self, obs_space, observations):
        obs = np.zeros(
            (self._env.num_envs,) + obs_space.shape, dtype=obs_space.dtype.name
        )
        obs[:] = observations
        return {"obs": obs}

    def _step(self, ep_infos):
        actions, post_actions, actionvalue_avg = self._agent_proxy.evaluate(self._obs)
        next_obs, rewards, dones, infos = self._env.step(post_actions)

        if self._render:
            self._env.venv.render_idx(0)
            # self.env.venv.render()  # TODO, we should really be calling this.

        datas_1 = {}
        datas_2 = {}
        for id, (o, a, r, d) in enumerate(
            zip(self._obs["obs"], actions, rewards, dones)
        ):
            datas_1[id + self._generations[id] * self._n_steps] = {
                "obs": o[None][0],
                "actions": a,
                "rewards": r,
            }
            if d:
                datas_2[id + self._generations[id] * self._n_steps] = {
                    "obs": next_obs[id][None][0]
                }
                self._generations[id] += 1

        self._agent_proxy.push_training_data(
            datas_1,
            datas_2,
        )

        completed_episodes = 0
        for info in infos:
            # TODO[tsolberg]: if we want to support /instant/ in collector-based
            # games, this should probably use an 'agent-metrics' key as well,
            # but all games have to change then.
            maybe_ep_info = info.get("episode")
            self._last_environment_rewards.extend(rewards)
            if maybe_ep_info is not None:
                completed_episodes += 1
                ep_infos["agent_metrics"].append(
                    {f"episode/{k}": v for k, v in maybe_ep_info.items()}
                )

        self._obs = {"obs": next_obs}
        return next_obs[0].shape[0], completed_episodes

    def collect_data(self):
        """Collect a single rollout"""
        ep_infos = {"agent_metrics": []}

        inference_steps = 0
        completed_episodes = 0
        for _ in range(self._n_steps):
            new_inference_steps, new_completed_episodes = self._step(ep_infos)
            inference_steps += new_inference_steps
            completed_episodes += new_completed_episodes

        self._rollouts += 1
        ep_infos["episode/completed"] = completed_episodes
        ep_infos["rollout/completed"] = self._rollouts
        ep_infos["env/reward"] = sum(self._last_environment_rewards) / len(
            self._last_environment_rewards
        )
        self._agent_proxy.report(ep_infos, {})

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
        _actions, post_actions, _actionvalue_avg = self._agent_proxy.evaluate(self._obs)
        self._env.step(post_actions)

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

    def __init__(self, env: gym.vector.VectorEnv, agent_proxy, render=True, bp_steps_per_inf=10, warmup_steps=0):
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
