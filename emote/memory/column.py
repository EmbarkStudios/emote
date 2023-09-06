"""

"""

from dataclasses import dataclass
from typing import Tuple, Type

import numpy as np

from emote.memory.storage import VirtualStorage


@dataclass
class Column:
    """A typed column for data storage."""

    name: str
    """The name of the column"""

    shape: Tuple[int]
    dtype: type

    def state(self):
        dtype_name = ""
        if hasattr(self.dtype, "name"):
            dtype_name = self.dtype.name
        else:
            dtype_name = self.dtype.__name__

        return {
            "shape": self.shape,
            "dtype": dtype_name,
        }

    def load_state(self, config):
        self.shape = config["shape"]

        dtype_name = config["dtype"]
        if dtype_name == "bool":
            self.dtype = bool

        else:
            if hasattr(np, dtype_name):
                self.dtype = getattr(np, dtype_name)
            else:
                self.dtype = np.dtype(dtype_name)


@dataclass
class TagColumn(Column):
    """A typed column for tag storage."""

    pass


@dataclass
class VirtualColumn(Column):
    """A column providing fake or transformed data via Mapper"""

    target_name: str
    mapper: Type[VirtualStorage]

    def state(self):
        config = super().state()
        config["target_name"] = self.target_name
        # TODO: add mapper configuration
        return config

    def load_state(self, config):
        super().load_state(config)
        self.target_name = config["target_name"]
