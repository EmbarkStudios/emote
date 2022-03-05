from itertools import count
import numpy as np
import gym.spaces

from gym.vector import VectorEnvWrapper, VectorEnv
from emote.typing import EpisodeState, HiveObservation, AgentId, HiveResponse
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


class HiveGymWrapper(VectorEnvWrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self._next_agent = count()
        self._agent_ids: list[AgentId] = [
            next(self._next_agent) for i in range(self.num_envs)
        ]
        assert isinstance(env.single_observation_space, gym.spaces.Box)
        os: gym.spaces.Box = env.single_observation_space
        self.hive_space = MDPSpace(
            BoxSpace(np.float32, (1,)),
            BoxSpace(env.single_action_space.dtype, env.single_action_space.shape),
            DictSpace({"obs": BoxSpace(os.dtype, os.shape)}),
        )

    def render(self):
        self.env.envs[0].render()

    def hive_step(
        self, actions: dict[AgentId, HiveResponse]
    ) -> dict[AgentId, HiveObservation]:
        batched_actions = np.stack(
            [actions[agent].list_data["actions"] for agent in self._agent_ids]
        )
        self.step_async(batched_actions)
        next_obs, rewards, dones, info = super().step_wait()
        new_agents = []
        results = {}
        for env_id, done in enumerate(dones):
            if done:
                results[self._agent_ids[env_id]] = HiveObservation(
                    episode_state=EpisodeState.TERMINAL,
                    array_data={"obs": next_obs[env_id]},
                    rewards={"reward": rewards[env_id]},
                )
                new_agent = next(self._next_agent)
                results[new_agent] = HiveObservation(
                    episode_state=EpisodeState.INITIAL,
                    array_data={"obs": next_obs[env_id]},
                    rewards={"reward": 0.0},
                )
                new_agents.append(new_agent)
                self._agent_ids[env_id] = new_agent

        results.update(
            {
                agent_id: HiveObservation(
                    episode_state=EpisodeState.RUNNING,
                    array_data={"obs": next_obs[env_id]},
                    rewards={"reward": rewards[env_id]},
                )
                for env_id, agent_id in enumerate(self._agent_ids)
                if agent_id not in new_agents
            }
        )
        return results

    def hive_reset(self) -> dict[AgentId, HiveObservation]:
        self._agent_ids = [next(self._next_agent) for i in range(self.num_envs)]
        self.reset_async()
        obs = self.reset_wait()
        return {
            agent_id: HiveObservation(
                episode_state=EpisodeState.INITIAL,
                array_data={"obs": obs[i]},
                rewards={"reward": 0.0},
            )
            for i, agent_id in enumerate(self._agent_ids)
        }
