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
        return {
            "shape": self.shape,
            "dtype": self.dtype.__name__,
        }

    def configure(self, config):
        self.shape = config["shape"]

        dtype_name = config["dtype"]
        if dtype_name == "bool":
            self.dtype = bool

        else:
            self.dtype = getattr(np, dtype_name)


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
