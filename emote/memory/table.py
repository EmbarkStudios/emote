from __future__ import annotations

import enum
import json
import logging
import os
import stat
import zipfile

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


logger = logging.getLogger(__name__)


class TableSerializationVersion(enum.Enum):
    """The version of the memory serialization format."""

    Legacy = 0
    """The legacy memory table format using pickling, which leads to portability issues and risks
    when refactoring."""

    V1 = 1
    """Memory table format using a zip file with a JSON metadata file and raw numpy data files. Note
    that this version only restores data, but will not affect the types of ejectors, adaptors, and
    so on."""

    LATEST = V1


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

    def store(
        self,
        path: str,
        version: TableSerializationVersion = TableSerializationVersion.LATEST,
    ) -> bool:
        """Persist the whole table and all metadata into the designated name"""
        ...

    def restore(
        self, path: str, override_version: TableSerializationVersion | None = None
    ) -> bool:
        """Restore the data table from the provided path. This also clears the data stores."""
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

    def resize(self, new_size):
        with self._lock:
            if new_size < self._maxlen:
                raise ValueError(
                    f"The new memory size {new_size} is smaller than the current size of the memory "
                    f"({self._maxlen}). Shrinking the memory is not supported"
                )
            self._maxlen = new_size

    def clear(self):
        """Clear and reset all data"""
        with self._lock:
            self._clear()

    def _clear(self):
        """Clear and reset all data"""

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
            f"and an error occurred when slicing the range {slice_begin}..{slice_end}"
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

                    except KeyError as err:
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
        with self._timers.scope("add_sequence"):
            with self._lock:
                self._add_sequence_internal(identity, sequence)

    def _add_sequence_internal(self, identity: int, sequence: dict):
        """add a fully terminated sequence to the memory"""
        sequence_length = len(sequence[self._length_key])

        # unsigned extend: all Ids that are added as sequences must be positive int64 values
        if identity < 0:
            identity += 2**64

            # Shrink before writing to make sure we don't overflow the storages

        with self._timers.scope("add_sequence_inner"):
            size_after_add = self._internal_size() + sequence_length
            if size_after_add > self._maxlen:
                self._eject_count(size_after_add - self._maxlen)

            for name, value in sequence.items():
                self._data[name][identity] = np.array(
                    value, dtype=self._columns[name].dtype
                ).reshape(-1, *self._columns[name].shape)

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

    def _serialize(self, path: str) -> bool:
        from atomicwrites import atomic_write

        with self._lock:
            with atomic_write(f"{path}.zip", overwrite=True, mode="wb") as tmp:
                with zipfile.ZipFile(tmp, "a") as zip_:
                    with zip_.open("version", "w") as version:
                        version_int = TableSerializationVersion.V1.value
                        version.write(str(version_int).encode("utf-8"))

                    parts = {
                        "ejector_type": self._ejector.__class__.__name__,
                        "sampler_type": self._sampler.__class__.__name__,
                        "length_key": self._length_key,
                        "maxlen": self._maxlen,
                        "ids": list(self._lengths.keys()),
                    }

                    ejector_state = self._ejector.state()
                    if ejector_state is not None:
                        parts["ejector_state"] = ejector_state

                    sampler_state = self._sampler.state()
                    if sampler_state is not None:
                        parts["sampler_state"] = sampler_state

                    parts["columns"] = [
                        (name, column.__class__.__name__, column.configuration())
                        for name, column in self._columns.items()
                    ]

                    output_ranges = {}
                    output_data = {}

                    for key, store in self._data.items():
                        ranges = []
                        merged_data = []

                        if isinstance(store, VirtualStorage):
                            continue

                        for identity, data in store.items():
                            ranges.append(
                                (
                                    identity,
                                    len(merged_data),
                                    len(data),
                                )
                            )
                            merged_data.extend(data)

                        output_data[key] = np.stack(merged_data)
                        output_ranges[key] = ranges

                    parts["part_keys"] = list(output_data.keys())

                    with zip_.open("configuration.json", "w", force_zip64=True) as f:
                        json_data = json.dumps(parts)
                        f.write(json_data.encode("utf-8"))

                    for key, data in output_data.items():
                        with zip_.open(f"{key}.ranges.npy", "w", force_zip64=True) as f:
                            np.save(f, output_ranges[key], allow_pickle=False)

                        with zip_.open(f"{key}.npy", "w", force_zip64=True) as npz:
                            np.save(npz, data, allow_pickle=False)

            os.chmod(
                f"{path}.zip", stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            )

    def _deserialize(self, zip_: "zipfile.ZipFile") -> bool:
        """Restore the data table from the provided path. This currently implies a "clear" of the data stores."""

        with self._lock:
            self._clear()

            self._ejector.begin_simple_import()
            self._sampler.begin_simple_import()

            with zip_.open("configuration.json", "r") as f:
                config = json.load(f)

            ejector_type = config["ejector_type"]

            if ejector_type != self._ejector.__class__.__name__:
                logger.warning(
                    f"Deserializing memory with ejector type {ejector_type}, but "
                    f"memory is configured with ejector type "
                    f"{self._ejector.__class__.__name__}. This may lead to "
                    f"unexpected behavior."
                )

            if "ejector_state" in config:
                self._ejector.load_state(config["ejector_state"])

            sampler_type = config["sampler_type"]

            if sampler_type != self._sampler.__class__.__name__:
                logger.warning(
                    f"Deserializing memory with sampler type {sampler_type}, but "
                    f"memory is configured with sampler type "
                    f"{self._sampler.__class__.__name__}. This may lead to "
                    f"unexpected behavior."
                )

            if "sampler_state" in config:
                self._sampler.load_state(config["sampler_state"])

            if self._length_key != config["length_key"]:
                logger.warning(
                    f"Deserializing memory with length key {config['length_key']}, "
                    f"but memory is configured with length key "
                    f"{self._length_key}. This may lead to unexpected behavior."
                )

            if self._maxlen != config["maxlen"]:
                logger.warning(
                    f"Deserializing memory with maxlen {config['maxlen']}, "
                    f"but memory is configured with maxlen "
                    f"{self._maxlen}. This may lead to unexpected behavior."
                )

            for name, column_type, column_config in config["columns"]:
                if name not in self._columns:
                    logger.warning(
                        f"Deserializing memory with column {name}, "
                        f"but memory is configured without column "
                        f"{name}. This may lead to unexpected behavior."
                    )
                    continue

                if column_type != self._columns[name].__class__.__name__:
                    logger.warning(
                        f"Deserializing memory with column {name} of type "
                        f"{column_type}, but memory is configured with column "
                        f"{name} of type {self._columns[name].__class__.__name__}. "
                        f"This may lead to unexpected behavior."
                    )
                    continue

                self._columns[name].configure(column_config)

            loaded_data = {}
            ranges = {}
            for key in config["part_keys"]:
                with zip_.open(f"{key}.ranges.npy", "r") as f:
                    tuplized = np.load(f, allow_pickle=False)

                    ranges[key] = {
                        identity: (start, size) for identity, start, size in tuplized
                    }

                with zip_.open(f"{key}.npy", "r") as npz:
                    loaded_data[key] = np.load(npz, allow_pickle=False)

            # we reassemble sequences and store them in the memory
            for identity in config["ids"]:
                reassembled = {}

                for key, data in ranges.items():
                    (start, size) = data[identity]
                    end = start + size
                    reassembled[key] = loaded_data[key][start:end]

                self._add_sequence_internal(identity, reassembled)

            self._sampler.end_simple_import()
            self._ejector.end_simple_import()

    def _store_legacy(self, path: str) -> bool:
        """Persist the whole table and all metadata into the designated name"""

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

                        cloudpickle.dump(parts, data_file, protocol=4)

                    for key, data in self._data.items():
                        if isinstance(data, VirtualStorage):
                            continue

                        with zip_.open(f"{key}.npy", "w", force_zip64=True) as npz:
                            np.save(npz, data)

            os.chmod(
                f"{path}.zip", stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
            )

    def _restore_legacy(self, zip_: zipfile.ZipFile) -> bool:
        """Restore the data table from the provided path. This currently implies a "clear" of the data stores."""

        import cloudpickle

        with self._lock:
            with zip_.open("data.pickle", "r") as data_file:
                parts = cloudpickle.load(data_file)
                self._ejector = parts["ejector"]
                self._sampler = parts["sampler"]
                self._length_key = parts["length_key"]
                self._maxlen = parts["maxlen"]
                self._columns = parts["columns"]
                self._lengths = parts["lengths"]
                self._filled = parts["filled"]

            for key, data in self._data.items():
                if isinstance(data, VirtualStorage):
                    continue

                with zip_.open(f"{key}.npy", "r") as npz:
                    loaded = np.load(npz, allow_pickle=True).item(0)
                    for d, v in loaded.items():
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

    def store(
        self,
        path: str,
        version: TableSerializationVersion = TableSerializationVersion.LATEST,
    ) -> bool:
        """Persist the whole table and all metadata into the designated name.

        :param path: The path to store the data to.
        :param use_legacy_format: Whether to use the legacy format for storing the data.
        """

        if version is None:
            version = TableSerializationVersion.LATEST

        if version == TableSerializationVersion.Legacy:
            return self._store_legacy(path)

        elif version == TableSerializationVersion.V1:
            return self._serialize(path)

        else:
            raise ValueError(f"Unknown serialization version {version}")

    def restore(
        self, path: str, override_version: TableSerializationVersion | None = None
    ) -> bool:
        with zipfile.ZipFile(f"{path}.zip", "r") as zip_:
            version = TableSerializationVersion.LATEST
            if override_version is not None:
                version = override_version
            elif "version" in zip_.namelist():
                with zip_.open("version", "r") as version_file:
                    version_int = int(version_file.read())
                    version = TableSerializationVersion(version_int)

            else:
                version = TableSerializationVersion.Legacy

            if version == TableSerializationVersion.Legacy:
                return self._restore_legacy(zip_)

            elif version == TableSerializationVersion.V1:
                return self._deserialize(zip_)

            else:
                raise ValueError(f"Unknown serialization version {version}")
