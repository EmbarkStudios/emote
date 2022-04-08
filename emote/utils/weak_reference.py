"""
A class that contains a typed weak reference.
"""

from typing import Generic, TypeVar
from weakref import ReferenceType


T = TypeVar("T")


class WeakReference(ReferenceType, Generic[T]):
    """A typed weak reference"""
