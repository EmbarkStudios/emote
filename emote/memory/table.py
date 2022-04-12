"""Example implementation for how a more modular memory pattern might look"""


from threading import Lock
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from ..utils.timed_call import BlockTimers
from .adaptors import Adaptor
from .column import Column, TagColumn, VirtualColumn
from .core_types import SampleResult
from .storage import BaseStorage, TagStorage, VirtualStorage
from .strategy import EjectionStrategy, SampleStrategy


class Table(Protocol):

    adaptors: List[Adaptor]

    def sample(self, count: int, sequence_length: int) -> SampleResult:
        """sample COUNT traces from the memory, each consisting of SEQUENCE_LENGTH
        frames. The data is transposed in a SoA fashion (since this is
        both easier to store and easier to consume).
        """
        ...

    def size(self) -> int:
        """query the number of elements currently in the memory"""
        ...

    def full(self) -> bool:
        """query whether the memory is filled"""
        ...

    def add_sequence(self, identity: int, sequence):
        """add a fully terminated sequence to the memory"""
        ...

    def store(self, path: str) -> bool:
        """Persist the whole table and all metadata into the designated name"""
        ...

    def restore(self, path: str) -> bool:
        """Restore the data table from the provided path. This currently implies a "clear" of the data stores."""
        ...


class ArrayTable:
    def __init__(
        self,
        *,
        columns: Sequence[Column],
        maxlen: int,
        sampler: SampleStrategy,
        ejector: EjectionStrategy,
        length_key="actions",
        adaptors: Optional[Adaptor] = None,
        device: torch.device,
    ):
        """Create the table with the specified configuration"""
        self._sampler = sampler
        self._ejector = ejector
        self._length_key = length_key
        self._maxlen = maxlen
        self._columns = {column.name: column for column in columns}
        self._lock = Lock()
        self.adaptors = adaptors if adaptors else []
        self._device = device

        self.clear()

    def clear(self):
        """Clear and reset all data"""
        with self._lock:
            self._data = {}

            self._lengths = {}
            self._total_length = 0
            self._filled = False
            self._timers = BlockTimers()

            for column in self._columns.values():
                if isinstance(column, VirtualColumn):
                    self._data[column.name] = column.mapper(
                        self._data[column.target_name], column.shape, column.dtype
                    )

                elif isinstance(column, TagColumn):
                    self._data[column.name] = TagStorage(column.shape, column.dtype)

                else:
                    self._data[column.name] = BaseStorage(column.shape, column.dtype)

    ################################################################################

    def _diagnostic_broadcast_error(
        self,
        err: Exception,
        key: str,
        episode_id: int,
        slice_begin: int,
        slice_end: int,
    ):
        """Assumptions: This is called while holding all the data lock"""

        lines = [f"Caught ValueError ({err}) when sampling memory for key {key}"]
        lines.append(f"\nFor episode id {episode_id}, the following data was found: ")
        for key, store in self._data.items():
            lines.append(f"\t{key} -> {store[episode_id].shape}")

        lines.append(
            f"and an error occured when slicing the range {slice_begin}..{slice_end}"
        )

        raise ValueError("\n".join(lines))

    ################################################################################

    def _execute_gather(
        self, count: int, sequence_length: int, sample_points: List[Tuple[int]]
    ):
        with self._timers.scope("gather"):
            out = {}
            for key, store in self._data.items():
                local_seq_length = store.sequence_length_transform(sequence_length)
                output_store = store.get_empty_storage(count, local_seq_length)
                idx = 0
                next_idx = idx + local_seq_length
                for identity, start, end in sample_points:
                    try:
                        output_store[idx:next_idx] = store[identity][start:end]
                        idx = next_idx

                    except ValueError as err:
                        self._diagnostic_broadcast_error(err, key, identity, start, end)

                    next_idx += local_seq_length

                out[key] = torch.tensor(output_store).to(self._device)

        return out

    def sample(self, count: int, sequence_length: int) -> SampleResult:
        """sample COUNT traces from the memory, each consisting of SEQUENCE_LENGTH
        transitions. The transitions are returned in a SoA fashion (since this is both
        easier to store and easier to consume)"""

        with self._lock:
            with self._timers.scope("points"):
                sample_points = self._sampler.sample(count, sequence_length)

            result = self._execute_gather(count, sequence_length, sample_points)
        for adaptor in self.adaptors:
            result = adaptor(result, count, sequence_length)
        return result

    def size(self) -> int:
        """query the number of elements currently in the memory"""
        with self._lock:
            return self._internal_size()

    def _internal_size(self) -> int:
        return self._total_length

    def full(self) -> bool:
        """Returns true if the memory has reached saturation, e.g., where new adds may
        cause ejection.

        .. warning:: This does not necessarily mean that `size() == maxlen`, as
           we store and eject full sequences. The memory only guarantees we will
           have *fewer* samples than maxlen.

        """
        with self._lock:
            return self._filled

    def add_sequence(self, identity: int, sequence: dict):
        """add a fully terminated sequence to the memory"""
        sequence_length = len(sequence[self._length_key])

        # unsigned extend: all Ids that are added as sequences must be positive int64 values
        if identity < 0:
            identity += 2**64

        # Shrink before writing to make sure we don't overflow the storages
        with self._timers.scope("add_sequence"):
            with self._lock:
                with self._timers.scope("add_sequence_inner"):
                    size_after_add = self._internal_size() + sequence_length
                    if size_after_add > self._maxlen:
                        self._eject_count(size_after_add - self._maxlen)

                    for name, value in sequence.items():
                        try:
                            # TODO add datatype type to memory? np/torch
                            dtype = self._columns[name].dtype
                            if dtype in [torch.float32]:
                                self._data[name][identity] = torch.stack(value).view(
                                    -1, *self._columns[name].shape
                                )
                            else:
                                self._data[name][identity] = np.array(
                                    value, dtype=dtype
                                ).reshape(-1, *self._columns[name].shape)
                        except:
                            print("foo")

                    self._total_length += sequence_length
                    self._lengths[identity] = sequence_length
                    self._sampler.track(identity, sequence_length)
                    self._ejector.track(identity, sequence_length)

    def _eject_count(self, count: int):
        """Request ejection of *at least* the specified number of transitions"""
        self._filled = True

        identities = self._ejector.sample(count)

        for to_eject in identities:
            for storage in self._data.values():
                del storage[to_eject]

            self._total_length -= self._lengths[to_eject]
            del self._lengths[to_eject]
            self._sampler.forget(to_eject)
            self._ejector.forget(to_eject)

    def store(self, path: str) -> bool:
        """Persist the whole table and all metadata into the designated name"""
        import zipfile

        import cloudpickle

        from atomicwrites import atomic_write

        with self._lock:
            with atomic_write(f"{path}.zip", overwrite=True, mode="wb") as tmp:
                with zipfile.ZipFile(tmp, "a") as zip_:
                    with zip_.open("data.pickle", "w", force_zip64=True) as data_file:
                        parts = {
                            "ejector": self._ejector,
                            "sampler": self._sampler,
                            "length_key": self._length_key,
                            "maxlen": self._maxlen,
                            "columns": self._columns,
                            "lengths": self._lengths,
                            "filled": self._filled,
                        }

                        cloudpickle.dump(parts, data_file)

                    for (key, data) in self._data.items():
                        if isinstance(data, VirtualStorage):
                            continue

                        with zip_.open(f"{key}.npy", "w", force_zip64=True) as npz:
                            np.save(npz, data)

    def restore(self, path: str) -> bool:
        """Restore the data table from the provided path. This currently implies a "clear" of the data stores."""
        import zipfile

        import cloudpickle

        with self._lock:
            with zipfile.ZipFile(f"{path}.zip", "r") as zip_:
                with zip_.open("data.pickle", "r") as data_file:
                    parts = cloudpickle.load(data_file)
                    self._ejector = parts["ejector"]
                    self._sampler = parts["sampler"]
                    self._length_key = parts["length_key"]
                    self._maxlen = parts["maxlen"]
                    self._columns = parts["columns"]
                    self._lengths = parts["lengths"]
                    self._filled = parts["filled"]

                for (key, data) in self._data.items():
                    if isinstance(data, VirtualStorage):
                        continue

                    with zip_.open(f"{key}.npy", "r") as npz:
                        loaded = np.load(npz, allow_pickle=True).item(0)
                        for (d, v) in loaded.items():
                            data[d] = v

            for column in self._columns.values():
                if isinstance(column, VirtualColumn):
                    self._data[column.name] = column.mapper(
                        self._data[column.target_name], column.shape, column.dtype
                    )

        self._lengths = {-abs(k) - 1: v for (k, v) in self._lengths.items() if k >= 0}
        self._total_length = sum(self._lengths.values())
        self._sampler.post_import()
        self._ejector.post_import()
        for column_store in self._data.values():
            column_store.post_import()
