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

    def configuration(self):
        dtype_name = ""
        if hasattr(self.dtype, "name"):
            dtype_name = self.dtype.name
        else:
            dtype_name = self.dtype.__name__

        return {
            "shape": self.shape,
            "dtype": dtype_name,
        }

    def configure(self, config):
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

    def configuration(self):
        config = super().configuration()
        config["target_name"] = self.target_name
        # TODO: add mapper configuration
        return config

    def configure(self, config):
        super().configure(config)
        self.target_name = config["target_name"]
