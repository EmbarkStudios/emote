import threading

from typing import Generic, TypeVar


T = TypeVar("T")


class LockedResource(Generic[T]):
    """Context manager for a lock and a resource, only giving access to the resource
    when locked. Works well when paired with :class:`empyc.types.Ref` for primitive
    types as well.


    Usage:

    .. code::

        resource = LockedResource([])
        with resource as inner_list:
             inner_list.append(1)

    """

    def __init__(self, data: T):
        """Create a new LockedResource, with the provided data.

        :param data: The data to lock"""
        self._lock = threading.Lock()
        self._data = data

    def __enter__(self) -> T:
        """Enter the locked context and retrieve the data.

        :returns: The underlying data object"""
        self._lock.acquire()
        return self._data

    def __exit__(self, _1, _2, _3):
        """Exit the locked context.

        .. note::

           Will propagate any errors occurring inside the locked scope.
        """
        self._lock.release()

    def swap(self, new_resource: T) -> T:
        """Replace the contained resource with the provided new resource, returning the
        previous resource. This operation is atomic.

        :param new_resource: The resource to lock after the swap
        :returns: The previously guarded data"""
        with self._lock:
            res = self._data
            self._data = new_resource

            return res


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
        """Increments the integer and returns the previous value"""
        with self._lock:
            original = self._value
            self._value += value
        return original
