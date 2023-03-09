from itertools import count
from typing import Dict, List

import gymnasium.spaces
import numpy as np

from gymnasium.vector import VectorEnv, VectorEnvWrapper

from emote.typing import AgentId, DictObservation, DictResponse, EpisodeState
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


class DictGymWrapper(VectorEnvWrapper):
    def __init__(self, env: VectorEnv):
        super().__init__(env)
        self._next_agent = count()
        self._agent_ids: List[AgentId] = [
            next(self._next_agent) for i in range(self.num_envs)
        ]
        self._episode_rewards: List[float] = [0.0 for i in range(self.num_envs)]
        assert isinstance(env.single_observation_space, gymnasium.spaces.Box)
        os: gymnasium.spaces.Box = env.single_observation_space
        if len(env.single_action_space.shape) > 0:
            action_space_shape = env.single_action_space.shape
        else:
            action_space_shape = (1,)
        self.dict_space = MDPSpace(
            BoxSpace(np.float32, (1,)),
            BoxSpace(env.single_action_space.dtype, env.single_action_space.shape),
            DictSpace({"obs": BoxSpace(os.dtype, os.shape)}),
        )

    def render(self):
        self.env.envs[0].render()

    def dict_step(
        self, actions: Dict[AgentId, DictResponse]
    ) -> Dict[AgentId, DictObservation]:
        batched_actions = np.stack(
            [actions[agent].list_data["actions"] for agent in self._agent_ids]
        )
        self.step_async(batched_actions)
        next_obs, rewards, dones, truncated, info = super().step_wait()
        new_agents = []
        results = {}
        completed_episode_rewards = []

        for env_id, reward in enumerate(rewards):
            self._episode_rewards[env_id] += reward

        for env_id, done in enumerate(dones):
            if done:
                results[self._agent_ids[env_id]] = DictObservation(
                    episode_state=EpisodeState.TERMINAL,
                    array_data={"obs": next_obs[env_id]},
                    rewards={"reward": rewards[env_id]},
                )
                new_agent = next(self._next_agent)
                results[new_agent] = DictObservation(
                    episode_state=EpisodeState.INITIAL,
                    array_data={"obs": next_obs[env_id]},
                    rewards={"reward": None},
                )
                new_agents.append(new_agent)
                completed_episode_rewards.append(self._episode_rewards[env_id])
                self._agent_ids[env_id] = new_agent
                self._episode_rewards[env_id] = 0.0

        results.update(
            {
                agent_id: DictObservation(
                    episode_state=EpisodeState.RUNNING,
                    array_data={"obs": next_obs[env_id]},
                    rewards={"reward": rewards[env_id]},
                )
                for env_id, agent_id in enumerate(self._agent_ids)
                if agent_id not in new_agents
            }
        )

        ep_info = {}
        if len(completed_episode_rewards) > 0:
            ep_info["reward"] = sum(completed_episode_rewards) / len(
                completed_episode_rewards
            )

        return results, ep_info

    def dict_reset(self) -> Dict[AgentId, DictObservation]:
        self._agent_ids = [next(self._next_agent) for i in range(self.num_envs)]
        self.reset_async()
        obs = self.reset_wait()
        return {
            agent_id: DictObservation(
                episode_state=EpisodeState.INITIAL,
                array_data={"obs": obs[0][i]},
                rewards={"reward": None},
            )
            for i, agent_id in enumerate(self._agent_ids)
        }
