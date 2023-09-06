from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from emote.memory.memory import TableMemoryProxy
from emote.memory.table import Table
from emote.typing import AgentId, DictObservation, DictResponse


class MemoryProxyWithEncoder(TableMemoryProxy):
    def __init__(
        self,
        table: Table,
        encoder: nn.Module,
        minimum_length_threshold: Optional[int] = None,
        use_terminal: bool = False,
        input_key: str = "obs",
        action_key: str = "actions",
    ):
        super().__init__(table, minimum_length_threshold, use_terminal)
        self.encoder = encoder
        self._input_key = input_key
        self._action_key = action_key

    def add(
        self,
        observations: Dict[AgentId, DictObservation],
        responses: Dict[AgentId, DictResponse],
    ):
        updated_responses = {}
        for agent_id, response in responses.items():
            actions = np.array(response.list_data[self._action_key])
            if np.size(actions) == 0:
                updated_responses.update({agent_id: response})
            else:
                actions = torch.from_numpy(actions).unsqueeze(dim=0).to(torch.float)
                obs = torch.from_numpy(
                    observations[agent_id].array_data[self._input_key]
                )
                obs = obs.unsqueeze(dim=0).to(torch.float)
                latent = self.encoder(actions, obs).squeeze().detach().cpu().numpy()
                new_response = DictResponse(
                    list_data={self._action_key: latent}, scalar_data={}
                )
                updated_responses.update({agent_id: new_response})
        super().add(observations, updated_responses)
