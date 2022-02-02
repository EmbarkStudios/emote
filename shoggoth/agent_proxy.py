"""
The AgentProxy is the interface between the agent in the game and the policy and memory used during training.
"""

from typing import Any, List, Mapping, Optional, Protocol, Tuple, Union

import numpy as np
from .sequence_builder import Observations, Terminals


MetadataList = Mapping[str, Union[float, int, List[float]]]
MetadataLists = Mapping[str, List[float]]

# Currently we adopt the convention of tupelizing lists with agents data, not the opposite.


class AgentProxy(Protocol):
    def evaluate(self, dict_obs: Mapping[str, np.ndarray]) -> Tuple[List[Any], ...]:
        ...

    def push_training_data(self, observations: Observations, terminals: Terminals):
        ...

    def report(self, metadata: MetadataList, metadata_lists: MetadataLists):
        ...

    def report_text(self, keyname: str, text: str):
        ...

    # def exported_checkpoints(self) -> Optional[ExportedCheckpointStorage]:
    #     ...

    # def create_checkpoint(self) -> StorageItem:
    #     ...

    def get_report(self, keys: List[str]) -> Tuple[MetadataList, MetadataLists]:
        ...

    def set_checkpoint_key_value(self, keyname: str, text: str):
        ...
