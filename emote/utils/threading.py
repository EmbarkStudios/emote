#!/usr/bin/env python3

"""Thread-related utilities and tools.

.. note::

   TODO
   ---

       * Make these generic over the locked resources
"""

import threading

from time import perf_counter
from typing import Any, Generic, TypeVar


T = TypeVar("T")


class LockedResource(Generic[T]):
    """Context manager for a lock and a resource, only giving access to the
    resource when locked. Works well when paired with :class:`empyc.types.Ref`
    for primitive types as well.

    Usage:

    .. code::

        resource = LockedResource([])
        with resource as inner_list:
             inner_list.append(1)
    """

    def __init__(self, data: T):
        """Create a new LockedResource, with the provided data.

        :param data: The data to lock
        """
        self._lock = threading.Lock()
        self._data = data

    def __enter__(self) -> T:
        """Enter the locked context and retrieve the data.

        :returns: The underlying data object
        """
        self._lock.acquire()
        return self._data

    def __exit__(self, _1, _2, _3):
        """Exit the locked context.

        .. note::

           Will propagate any errors occurring inside the locked scope.
        """
        self._lock.release()

    def swap(self, new_resource: T) -> T:
        """Replace the contained resource with the provided new resource,
        returning the previous resource. This operation is atomic.

        :param new_resource: The resource to lock after the swap
        :returns: The previously guarded data
        """
        with self._lock:
            res = self._data
            self._data = new_resource

            return res


class AtomicContainer:
    """Container that allows atomic ``set``, ``get``, ``take`` operations."""

    def __init__(self, initial_data: Any = None):
        self._data = initial_data
        self._lock = threading.Lock()

    def take(self) -> Any:
        with self._lock:
            value, self._data = self._data, None
            return value

    def read(self) -> Any:
        with self._lock:
            return self._data

    def set(self, value: Any):
        with self._lock:
            self._data = value


class AtomicInt:
    def __init__(self, value: int = 0):
        self._value = value
        self._lock = threading.Lock()

    def __iadd__(self, value: int):
        with self._lock:
            self._value += value
            return self

    def __isub__(self, value: int):
        with self._lock:
            self._value -= value
            return self

    def swap(self, value: int):
        with self._lock:
            current_value = self._value
            self._value = value
            return current_value

    def increment(self, value: int = 1):
        """Increments the integer and returns the previous value."""
        with self._lock:
            original = self._value
            self._value += value
        return original


class TracedLock:
    def __init__(self, lock_class=threading.Lock):
        self._lock = lock_class()
        self._current_scope_exec_start = None
        self._accumulated_block_time = 0.0
        self._accumulated_exec_time = 0.0

    def __enter__(self):
        try_lock_start = perf_counter()
        self._lock.acquire()
        now = perf_counter()
        self._accumulated_block_time += now - try_lock_start
        self._current_scope_exec_start = now
        return self

    def __exit__(self, *args, **kwargs):
        self._lock.release()
        self._accumulated_exec_time += perf_counter() - self._current_scope_exec_start


__all__ = ["LockedResource", "AtomicContainer", "AtomicInt", "TracedLock"]
