import onnx
import pytest
import torch

from gymnasium.vector import AsyncVectorEnv

from emote.proxies import GenericAgentProxy
from emote.extra.onnx_exporter import OnnxExporter
from emote.nn.gaussian_policy import GaussianMlpPolicy as Policy

from .gym import DictGymWrapper, HitTheMiddle


N_HIDDEN = 10


@pytest.fixture
def exporter(tmpdir):
    device = torch.device("cpu")
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))

    policy = Policy(2, 1, [N_HIDDEN, N_HIDDEN])

    input_keys = list(env.dict_space.state.spaces.keys())
    agent_proxy = GenericAgentProxy(policy, device, input_keys, True, ["actions"])

    exporter = OnnxExporter(
        agent_proxy,
        env.dict_space,
        True,
        tmpdir / "inference",
        50,
    )

    return exporter


def test_onnx_metadata_set(exporter):
    exporter.add_metadata("this is a key", "this is a value")
    exporter.add_metadata("this will be overridden", "oh no!")

    handle = exporter.export(
        {
            "this is another key": "this is another value",
            "this will be overridden": "oh yes!",
        }
    )

    with open(handle.filepath, "rb") as f:
        model = onnx.load_model(f, onnx.ModelProto)

    print(model.metadata_props)

    assert len(model.metadata_props) == 3
    assert model.metadata_props[0].key == "this is a key"
    assert model.metadata_props[0].value == "this is a value"

    assert model.metadata_props[1].key == "this will be overridden"
    assert model.metadata_props[1].value == "oh yes!"

    assert model.metadata_props[2].key == "this is another key"
    assert model.metadata_props[2].value == "this is another value"


def test_onnx_requires_str_key(exporter):
    with pytest.raises(TypeError):
        exporter.add_metadata(1, "this is a value")

    with pytest.raises(TypeError):
        exporter.export(
            {
                1: "this is another value",
            }
        )


def test_onnx_converts_value_to_str(exporter):
    exporter.add_metadata("this is a key", 1)

    handle = exporter.export(
        {
            "this is another key": 2,
        }
    )

    with open(handle.filepath, "rb") as f:
        model = onnx.load_model(f, onnx.ModelProto)

    assert model.metadata_props[0].key == "this is a key"
    assert model.metadata_props[0].value == "1"

    assert model.metadata_props[1].key == "this is another key"
    assert model.metadata_props[1].value == "2"
