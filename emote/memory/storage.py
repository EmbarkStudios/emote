"""

"""

from typing import Sequence, Tuple, Union

import numpy as np

from .core_types import Number


class BaseStorage(dict):
    """A simple dictionary-based storage with support for a temporary workspace for
    sampled data"""

    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = dtype
        self._temp_storage = None

    def get_empty_storage(self, count, length):
        """A workspace that can be reused to skip reallocating the same numpy buffer
        each time the memory is sampled. Will *not* work if the memory is
        sampled from multiple threads.
        """
        total_size = count * length
        if self._temp_storage is None or self._temp_storage.shape[0] < total_size:
            d = np.empty((total_size, *self._shape), self._dtype)
            self._temp_storage = d

        return self._temp_storage[:total_size]

    def sequence_length_transform(self, length):
        return length

    def post_import(self):
        # have to coalesce the list explicitly since we'll otherwise suffer from iterator invalidation
        invalid_ids = list(filter(lambda v: v < 0, self.keys()))

        # delete all imported negative ids
        for invalid_id in invalid_ids:
            del self[invalid_id]

        # make remaining ids negative
        remaining_ids = list(self.keys())
        for valid_id in remaining_ids:
            self[-abs(valid_id) - 1] = self[valid_id]
            del self[valid_id]


class TagStorage(dict):
    class TagProxy:
        __slots__ = ["value"]

        def __getitem__(self, key):
            return self.value[0]

        @property
        def shape(self):
            return (-1,)

    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = dtype
        self._temp_storage = None

    def get_empty_storage(self, count, length):
        """A workspace that can be reused to skip reallocating the same numpy buffer
        each time the memory is sampled. Will *not* work if the memory is
        sampled from multiple threads.
        """
        total_size = count * length
        if self._temp_storage is None or self._temp_storage.shape[0] < total_size:
            d = np.empty((total_size, *self._shape), self._dtype)
            self._temp_storage = d

        return self._temp_storage[:total_size]

    def sequence_length_transform(self, length):
        return 1

    def post_import(self):
        # have to coalesce the list explicitly since we'll otherwise suffer from iterator invalidation
        invalid_ids = list(filter(lambda v: v < 0, self.keys()))

        # delete all imported negative ids
        for invalid_id in invalid_ids:
            del self[invalid_id]

        # make remaining ids negative
        remaining_ids = list(self.keys())
        for valid_id in remaining_ids:
            self[-abs(valid_id) - 1] = self[valid_id]
            del self[valid_id]

    def __getitem__(self, key: int | Tuple[int, ...] | slice):
        episode = super().__getitem__(key)
        r = TagStorage.TagProxy()
        r.value = episode
        return r

    @property
    def shape(self):
        return (0,)


class VirtualStorage:
    """A virtual storage uses a simple storage to generate data"""

    def __init__(self, storage, shape, dtype):
        self._storage = storage
        self._shape = shape
        self._dtype = dtype
        self._temp_storage = None

    @property
    def shape(self):
        return self._storage.shape

    def __getitem__(self, key: int | Tuple[int, ...] | slice):
        pass

    def __setitem__(self, key: int | Tuple[int, ...] | slice, value: Sequence[Number]):
        pass

    def __delitem__(self, key: int | Tuple[int, ...] | slice):
        pass

    def sequence_length_transform(self, length):
        return length

    def get_empty_storage(self, count, length):
        total_size = count * length
        if self._temp_storage is None or self._temp_storage.shape[0] < total_size:
            d = np.empty((total_size, *self._shape), self._dtype)
            self._temp_storage = d

        return self._temp_storage[:total_size]

    def post_import(self):
        pass


class NextElementMapper(VirtualStorage):
    """Simple mapper that can be used to sample a specified one step over, which is
    useful to sample transitions for RL."""

    class Wrapper:
        def __init__(self, item):
            self._item = item

        def __getitem__(self, key):
            if isinstance(key, int):
                key += 1

            elif isinstance(key, tuple):
                key = tuple(k + 1 for k in key)

            elif isinstance(key, slice):
                key = slice(key.start + 1, key.stop + 1, key.step)

            return self._item[key]

        @property
        def shape(self):
            return self._item.shape

    class LastWrapper:
        def __init__(self, item):
            self._item = item

        def __getitem__(self, key):
            if isinstance(key, int):
                key += 1

            elif isinstance(key, tuple):
                key = tuple(k + 1 for k in key[-1:])

            elif isinstance(key, slice):
                step = key.step or 1
                key = slice(key.stop, key.stop + step, step)

            return self._item[key]

        @property
        def shape(self):
            return self._item.shape

    def __init__(self, storage, shape, dtype, only_last: bool = False):
        super().__init__(storage, shape, dtype)
        self._only_last = only_last
        self._wrapper = NextElementMapper.LastWrapper if only_last else NextElementMapper.Wrapper

    def __getitem__(self, key: int | Tuple[int, ...] | slice):
        return self._wrapper(self._storage[key])

    def sequence_length_transform(self, length):
        return 1 if self._only_last else length

    @staticmethod
    def with_only_last(storage, shape, dtype):
        return NextElementMapper(storage, shape, dtype, only_last=True)


class SyntheticDones(VirtualStorage):
    """Generates done or masks based on sequence length."""

    class Wrapper:
        def __init__(self, length, shape, dtype):
            self._max_idx = length - 1
            self._shape = shape
            self._dtype = dtype

        def __getitem__(self, key):
            if isinstance(key, int):
                return self._dtype(key == self._max_idx)

            elif isinstance(key, tuple):
                return tuple(self._dtype(k == self._max_idx) for k in key)

            elif isinstance(key, slice):
                return (
                    (np.arange(key.start, key.stop) == self._max_idx)
                    .reshape(-1, *self._shape)
                    .astype(self._dtype)
                )

        @property
        def shape(self):
            return (-1,)

    class MaskWrapper(Wrapper):
        def __getitem__(self, key):
            if isinstance(key, int):
                return self._dtype(1.0 - (key == self._max_idx))

            elif isinstance(key, tuple):
                return tuple(self._dtype(1.0 - (k == self._max_idx)) for k in key)

            elif isinstance(key, slice):
                v = SyntheticDones.Wrapper.__getitem__(self, key)
                return 1.0 - v

    def __init__(self, storage, shape, dtype, mask: bool = False):
        super().__init__(storage, shape, dtype)
        self._mask = mask

    def __getitem__(self, key: int | Tuple[int, ...] | slice):
        if self._mask:
            return SyntheticDones.MaskWrapper(len(self._storage[key]), self._shape, self._dtype)

        return SyntheticDones.Wrapper(len(self._storage[key]), self._shape, self._dtype)

    @staticmethod
    def as_mask(storage, shape, dtype):
        return SyntheticDones(storage, shape, dtype, mask=True)
