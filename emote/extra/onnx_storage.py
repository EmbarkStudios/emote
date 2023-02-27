from __future__ import annotations

import io
import logging
import time

from queue import Empty, Queue
from threading import Event
from typing import Mapping, Optional, Sequence

import onnx
import torch

from google.protobuf import text_format
from torch import nn

from emote.extra.crud_storage import CRUDStorage, StorageItem, StorageItemHandle
from emote.utils.timed_call import BlockTimers


class QueuedExport:
    def __init__(self, metadata: Optional[Mapping[str, str]]):
        self.metadata = metadata
        self.return_value = None
        self._event = Event()

    def process(self, storage: "OnnxExporter"):
        self.return_value = storage._export_onnx(self.metadata)
        self._event.set()

    def block_until_complete(self):
        self._event.wait()
        return self.return_value


def _get_version():
    try:
        from importlib.metadata import version

        return version("emote-rl")
    except:  # noqa
        return "unknown-version"


def _save_protobuf(path, message, as_text: bool = False):
    import os

    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(message))
    else:
        with open(path, "wb") as f:
            f.write(message.SerializeToString())


class OnnxExporter:
    """Handles onnx exports of a ML policy.

    Call `export` whenever you want to save an onnx version of the
    current model, or `export_threadsafe` if you're outside the
    training loop.

    Parameters:
    :param directory: path to the directory where the files should be created. If it does not exist
                      it will be created.
    :param prefix: all file names will have this prefix.

    """

    def __init__(
        self,
        policy: nn.Module,
        directory: str,
        input_shapes: list[tuple[str, tuple[int]]],
        output_shapes: list[tuple[str, tuple[int]]],
        prefix: str = "savedmodel_",
    ):
        self.policy = policy
        self.storage = CRUDStorage(directory, prefix, extension="onnx")
        self.queued_exports = Queue()
        self.export_counter = 0

        # Cache the version tag on startup.
        # It takes about 20ms to calculate
        self.version_tag = _get_version()
        self.scopes = BlockTimers()

        self.inputs = input_shapes
        self.outputs = output_shapes

    def process_pending_exports(self):
        """
        If you are using `export_threadsafe` the main thread must call
        this method regularly to make sure things are actually exported.
        """
        while self.queued_exports.qsize() > 0:
            try:
                item = self.queued_exports.get_nowait()
            except Empty:
                return
            item.process(self)

    def _trace(self):
        with self.scopes.scope("trace"):
            args = []

            for _, shape in self.inputs:
                args.append(torch.randn(1, *shape))

            self.policy.train(False)
            with torch.no_grad():
                with torch.jit.optimized_execution(True):
                    trace = torch.jit.trace(
                        self.policy,
                        args,
                        check_trace=True,
                        check_tolerance=1e-05,
                        strict=True,
                    )

            self.policy.train(True)

            with io.BytesIO() as f:
                torch.onnx.export(
                    model=trace,
                    args=args,
                    f=f,
                    input_names=list(map(lambda pair: pair[0], self.inputs)),
                    output_names=list(map(lambda pair: pair[0], self.outputs)),
                    dynamic_axes={
                        **{pair[0]: {0: "N"} for pair in self.inputs},
                        **{pair[0]: {0: "N"} for pair in self.outputs},
                    },
                    opset_version=13,
                )

                f.seek(0)
                model_proto = onnx.load_model(f, onnx.ModelProto)

            return model_proto

    def _export_onnx(self, metadata: Optional[Mapping[str, str]]) -> StorageItem:
        def save_inner(export_path: str):
            with self.scopes.scope("save"):
                t0 = time.time()
                model_proto = self._trace()
                model_version = self.export_counter
                self.export_counter += 1

                model_proto.producer_name = "emote"
                model_proto.domain = "dev.embark.ml"
                model_proto.producer_version = self.version_tag
                model_proto.model_version = model_version
                model_proto.doc_string = "exported via Emote checkpointer"

                if metadata is not None:
                    onnx.helper.set_model_props(model_proto, metadata)

                _save_protobuf(export_path, model_proto)

                t1 = time.time()
                logging.info(f"Exported onnx in {round((t1-t0)*1000)} ms")

        with self.scopes.scope("create"):
            return self.storage.create_with_saver(save_inner)

    def _export(self, metadata: Optional[Mapping[str, str]], sync: bool) -> StorageItem:
        # The actual onnx export needs to be done on the main thread.
        item = QueuedExport(metadata)
        self.queued_exports.put(item)
        if sync:
            # This will cause block_until_complete to never block
            # because the work will have been completed already.
            self.process_pending_exports()

        return item.block_until_complete()

    def export_threadsafe(self, metadata=None) -> StorageItem:
        """
        Same as `export`, but it can be called in threads other than the main thread.
        This method relies on the main thread calling `process_pending_exports` from time to time.
        You cannot call this method from the main thread. It will block indefinitely.
        """
        return self._export(metadata, False)

    def export(self, metadata=None) -> StorageItem:
        """
        Serializes a model to onnx and saves it to disk.
        This must only be called from the main thread.
        That is, the thread which has ownership over the model and that modifies it.
        This is usually the thread that has the training loop.
        """
        return self._export(metadata, True)

    def delete(self, handle: StorageItemHandle) -> bool:
        return self.storage.delete(handle)

    def get(self, handle: StorageItemHandle) -> bool:
        return self.storage.get(handle)

    def items(self) -> Sequence[StorageItem]:
        return self.storage.items()

    def latest(self) -> Optional[StorageItem]:
        return self.storage.latest()
