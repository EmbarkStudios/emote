"""
Generic CRUD-based storage on disk.
"""
from __future__ import annotations
import os

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Generic, Sequence, TypeVar

from emote.utils.threading import AtomicInt, LockedResource


T = TypeVar("T")


@dataclass(frozen=True)
class StorageItemHandle(Generic[T]):
    """
    A handle that represents a storage item.
    Can be safely exposed to users.
    Not cryptographically safe: handles are guessable.

    You can convert this handle from and to strings using `str(handle)` and
    `StorageItemHandle.from_string(string)`.
    """

    handle: int

    @staticmethod
    def from_string(value: str) -> "StorageItemHandle" | None:
        """
        Parses a handle from its string representation.
        Returns None if the handle is invalid.
        """
        try:
            return StorageItemHandle(int(value))
        except ValueError:
            return None

    def __str__(self):
        return str(self.handle)


@dataclass(frozen=True)
class StorageItem(Generic[T]):
    # A handle that represents this item.
    # Can be safely exposed to users.
    # Not cryptographically safe: handles are guessable.
    handle: StorageItemHandle[T]
    # When the file was created (in UTC)
    timestamp: datetime
    # Path to the file in the filesystem
    filepath: str


class CRUDStorage(Generic[T]):
    """
    Manages a set of files on disk in a simple CRUD way.
    All files will be stored to a single directory with a name on the format
    `{prefix}{timestamp}_{index}.{extension}`.

    This class is thread-safe.
    """

    def __init__(self, directory: str, prefix: str = "", extension: str = "bin"):
        assert len(extension) > 0
        assert "." not in extension, "Extension should not contain a dot"

        directory = os.path.abspath(directory)
        if os.path.exists(directory) and not os.path.isdir(directory):
            # Path exists, but it is not a directory
            ENOTDIR = 20
            raise os.error(ENOTDIR, os.strerror(ENOTDIR), directory)

        os.makedirs(directory, exist_ok=True)

        self._directory = directory
        self._filename_counter = AtomicInt(0)
        self._items: list[StorageItem[T]] = LockedResource([])
        self._extension = extension
        self._prefix = prefix

    def create_with_data(self, data: bytearray) -> StorageItem[T]:
        """Creates a new file with the given data"""

        def save(filepath):
            with open(filepath, "wb") as f:
                f.write(data)

        return self.create_with_saver(save)

    def create_from_filepath(self, filepath: str) -> StorageItem[T]:
        """
        Creates a new entry for an existing file.
        The file must already be in the directory that this storage manages.
        It does not need to conform to the naming convention that the CRUDStorage normally uses.
        """
        assert os.path.isfile(filepath), "File does not exist"
        if os.path.dirname(os.path.abspath(filepath)) != self._directory:
            raise Exception(
                f"Cannot add '{filepath}' to the storage because it                    "
                f" is not in the storage directory '{self._directory}'"
            )

        utcdate = datetime.utcnow()
        handle = StorageItemHandle(self._filename_counter.increment())
        item = StorageItem(timestamp=utcdate, filepath=filepath, handle=handle)

        with self._items as items:
            items.append(item)
        return item

    def create_with_saver(self, saver: Callable[[str], None]) -> StorageItem[T]:
        """
        Creates a new file by saving it via the provided function.
        The function will be called with the path at which the file should be saved.
        """
        if not os.path.isdir(self._directory):
            raise Exception(
                f"The storage directory ({self._directory}) has been deleted"
            )

        # Get the local time
        date = datetime.now()
        # Get the time in UTC. Converting between timezones is a bit annoying without external
        # libraries in python.
        utcdate = datetime.utcnow()

        # Try to find a valid filename.
        # In rare cases where files exist that we didn't know about we may have
        # to increment _filename_counter multiple times to find a valid filename.
        while True:
            # We need to use an atomic int here to ensure there is no race condition when
            # generating file paths.  We don't want it to be possible to call `create_with_saver`
            # from two threads at the same time and they both try to write to the same file path.
            handle = StorageItemHandle(self._filename_counter.increment())
            # The filename is formatted in local time for ease of use
            datestr = date.strftime(r"%Y-%m-%d_%H-%M")
            filename = f"{self._prefix}{datestr}_{handle}.{self._extension}"
            filepath = os.path.join(self._directory, filename)
            if not os.path.exists(filepath):
                break

        item = StorageItem(timestamp=utcdate, filepath=filepath, handle=handle)
        saver(item.filepath)
        assert os.path.isfile(
            item.filepath
        ), f"Saver did not save the data to the provided filepath {item.filepath}"
        with self._items as items:
            items.append(item)
        return item

    def update(self, handle: StorageItemHandle[T], data: bytearray):
        """
        Updates an existing file with the given contents
        """
        item = self.get(handle)
        assert item is not None, "Invalid handle"
        with open(item.filepath, "wb") as f:
            f.write(data)

    def items(self) -> Sequence[StorageItem[T]]:
        """
        :returns: a sequence of all files owned by this storage.
        """
        with self._items as items:
            # Return a copy of the list to ensure it can be safely handed to different threads
            # without the list potentially being modified while they are using it.
            return items[:]

    def delete(self, handle: StorageItemHandle[T]) -> bool:
        """
        Deletes an existing file owned by this storage.
        :returns: True if a file was deleted, and false if the file was not owned by this storage.
        :raises: Exception if this storage contains an entry for the file,
                 but it has been deleted on disk without going through the CRUDStorage.
        """
        with self._items as items:
            for item in items:
                if item.handle != handle:
                    continue

                items.remove(item)
                try:
                    os.remove(item.filepath)
                except FileNotFoundError as e:
                    raise Exception(
                        f"The file {item.filepath} has already been deleted            "
                        "                 without going through the CRUDStorage"
                    ) from e
                return True

        return False

    def get(self, handle: StorageItemHandle[T]) -> StorageItem[T] | None:
        """
        :returns: The storage item corresponding handle or None if it was not found
        """

        with self._items as items:
            # Slow, but this class is not expected to have to handle a large number of files
            return next((item for item in items if item.handle == handle), None)

    def latest(self) -> StorageItem[T] | None:
        """
        The last storage item that was added to the storage.
        If items have been deleted, this is the last item of the ones that remain.
        """
        with self._items as items:
            if items:
                return items[-1]

        return None
