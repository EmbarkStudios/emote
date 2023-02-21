import io
import warnings

import onnx
import torch

from torch import Tensor

from emote.utils.spaces import MDPSpace


class OnnxGeneratorMixin:
    HAS_INSERTED_FILTER = None

    def __init__(self, *args, spaces: MDPSpace, with_epsilon: bool = True, **kwargs):
        """
        Construct a new onnx exporter, configuring the API as above.
        Assumes the first dimension of each input and output is dynamic.
        """
        super().__init__(*args, **kwargs)

        # The pytorch onnx warns while trying to figure out the
        # forward signature of our module, but since this exporter
        # requires specifying it manually it'll be OK.
        if not OnnxGeneratorMixin.HAS_INSERTED_FILTER:
            OnnxGeneratorMixin.HAS_INSERTED_FILTER = True
            # This is caused by our old version of torch
            warnings.filterwarnings("ignore", "Skipping _decide_input_format.*")

            # https://github.com/pytorch/pytorch/issues/74799
            warnings.filterwarnings("ignore", "Model has no forward function")
        self.input_shapes = {k: v.shape for (k, v) in spaces.state.spaces.items()}

        if with_epsilon:
            self.input_shapes["epsilon"] = (*spaces.actions.shape,)

        self.output_shapes = {"actions": (*spaces.actions.shape,)}
        self.with_epsilon = with_epsilon

    def _gen_input_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        outputs = []

        for key in names:
            shape = self.input_shapes[key]
            outputs.append(torch.randn(1, *shape))

        return outputs

    def _generate_onnx(
        self, trace: torch.jit.ScriptFunction, args: list[Tensor], names: tuple[str]
    ):
        with io.BytesIO() as f:
            torch.onnx.export(
                model=trace,
                args=args,
                f=f,
                input_names=list(names),
                output_names=list(self.output_shapes.keys()),  # only one atm so ok
                dynamic_axes={
                    **{k: {0: "N"} for k in self.input_shapes.keys()},
                    **{k: {0: "N"} for k in self.output_shapes.keys()},
                },
                opset_version=13,
            )

            f.seek(0)
            model_proto = onnx.load_model(f, onnx.ModelProto)

        return model_proto
