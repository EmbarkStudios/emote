"""

"""

from dataclasses import dataclass
from typing import Tuple, Type

from emote.memory.storage import VirtualStorage


@dataclass
class Column:
    """A typed column for data storage."""

    name: str
    """The name of the column"""

    shape: tuple[int]
    dtype: type


@dataclass
class TagColumn(Column):
    """A typed column for tag storage."""

    pass


@dataclass
class VirtualColumn(Column):
    """A column providing fake or transformed data via Mapper"""

    target_name: str
    mapper: Type[VirtualStorage]
