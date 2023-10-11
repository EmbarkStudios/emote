"""
Proxies are bridges between the world the agent acts in and the algorithm training loop.
"""


from __future__ import annotations

from typing import Dict, Protocol

import numpy as np
import torch

from torch import nn

from emote.typing import AgentId, DictObservation, DictResponse, EpisodeState
from emote.utils.spaces import MDPSpace


class AgentProxy(Protocol):
    """The interface between the agent in the game and the network used during training."""

    def __call__(
        self,
        obserations: Dict[AgentId, DictObservation],
    ) -> Dict[AgentId, DictResponse]:
        """Take observations for the active agents and returns the relevant network output."""
        ...

    @property
    def policy(self) -> nn.Module:
        pass

    @property
    def input_names(self) -> tuple[str, ...]:
        ...

    @property
    def output_names(self) -> tuple[str, ...]:
        ...


class MemoryProxy(Protocol):
    """The interface between the agent in the game and the memory buffer the network trains from."""

    def add(
        self,
        observations: Dict[AgentId, DictObservation],
        responses: Dict[AgentId, DictResponse],
    ):
        """Store episodes in the memory buffer used for training.

        This is useful e.g. if the data collection is running from a checkpointed model running on
        another machine."""
        ...


class GenericAgentProxy(AgentProxy):
    """Observations are dicts that contain multiple input and output keys.

    For example, we might have a policy that takes in both "obs" and
    "goal" and outputs "actions". In order to be able to properly
    invoke the network it is the responsibility of this proxy to
    collate the inputs and decollate the outputs per agent.
    """

    def __init__(
        self,
        policy: nn.Module,
        device: torch.device,
        input_keys: tuple,
        output_keys: tuple,
        uses_logprobs: bool = True,
        spaces: MDPSpace | None = None,
    ):
        r"""Handle multi-input multi-output policy networks.

        Parameters:
            policy (nn.Module): The neural network policy that takes observations and returns actions.
            device (torch.device): The device to run the policy on.
            input_keys (tuple): Keys specifying what fields from the observation to pass to the policy.
            output_keys (tuple): Keys for the fields in the output dictionary that the policy is responsible for.
            spaces (MDPSpace, optional): A utility for managing observation and action spaces, for validation.
        """
        self._policy = policy
        self._end_states = [EpisodeState.TERMINAL, EpisodeState.INTERRUPTED]
        self.device = device
        self.input_keys = input_keys
        self.output_keys = output_keys
        self._spaces = spaces
        self._uses_logprobs = uses_logprobs

    def __call__(self, observations: dict[AgentId, DictObservation]) -> dict[AgentId, DictResponse]:
        """Runs the policy and returns the actions."""
        # The network takes observations of size batch x obs for each observation space.
        assert len(observations) > 0, "Observations must not be empty."

        active_agents = [
            agent_id
            for agent_id, obs in observations.items()
            if obs.episode_state not in self._end_states
        ]

        tensor_obs_list = [None] * len(self.input_keys)
        for input_key in self.input_keys:
            np_obs = np.array(
                [observations[agent_id].array_data[input_key] for agent_id in active_agents]
            )

            if self._spaces is not None:
                shape = (np_obs.shape[0],) + self._spaces.state.spaces[input_key].shape
                if shape != np_obs.shape:
                    np_obs = np.reshape(np_obs, shape)

            tensor_obs = torch.tensor(np_obs).to(self.device)
            index = self.input_keys.index(input_key)
            tensor_obs_list[index] = tensor_obs

        if self._uses_logprobs:
            outputs: tuple[any, ...] = self._policy(*tensor_obs_list)
            # we remove element 1 as we don't need the logprobs here
            outputs = outputs[0:1] + outputs[2:]
            outputs = {
                key: outputs[i].detach().cpu().numpy() for i, key in enumerate(self.output_keys)
            }
        else:
            outputs = self._policy(*tensor_obs_list)
            outputs = {key: outputs.detach().cpu().numpy() for key in self.output_keys}

        agent_data = [
            (agent_id, DictResponse(list_data={}, scalar_data={})) for agent_id in active_agents
        ]

        for i, (_, response) in enumerate(agent_data):
            for k, data in outputs.items():
                response.list_data[k] = data[i]

        return dict(agent_data)

    @property
    def input_names(self):
        return self.input_keys

    @property
    def output_names(self):
        return self.output_keys

    @property
    def policy(self):
        return self._policy
