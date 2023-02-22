"""

"""

import warnings

from emote.callback import Callback
from emote.extra.onnx_storage import OnnxExporter
from emote.proxies import AgentProxy
from emote.utils.spaces import MDPSpace


__HAS_INSERTED_FILTER = False


class OnnxExporterCallback(Callback):
    def __init__(self, exporter: OnnxExporter, interval: int):
        super().__init__(cycle=interval)
        self.exporter = exporter

    def end_cycle(self):
        self.exporter.process_pending_exports()
        self.exporter.export()


def make_onnx_exporter_bundle(
    agent_proxy: AgentProxy,
    spaces: MDPSpace,
    directory: str,
    export_interval: int,
    requires_epsilon: bool = False,
):
    global __HAS_INSERTED_FILTER
    if not __HAS_INSERTED_FILTER:
        __HAS_INSERTED_FILTER = True
        # This is caused by our old version of torch
        warnings.filterwarnings("ignore", "Skipping _decide_input_format.*")

        # https://github.com/pytorch/pytorch/issues/74799
        warnings.filterwarnings("ignore", "Model has no forward function")

    input_names = agent_proxy.input_names
    input_shapes = [(k, spaces.state.spaces[k].shape) for k in input_names]

    if requires_epsilon:
        input_names = (*input_names, "epsilon")
        input_shapes.append(
            (
                "epsilon",
                (*spaces.actions.shape,),
            )
        )

    output_shapes = [("actions", (*spaces.actions.shape,))]

    storage = OnnxExporter(
        agent_proxy.policy,
        directory,
        input_shapes,
        output_shapes,
    )

    return (
        storage,
        OnnxExporterCallback(storage, export_interval),
    )
