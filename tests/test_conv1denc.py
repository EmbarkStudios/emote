import pytest
import torch

from emote.nn.layers import Conv1dEncoder


def test_conv1denc():
    bsz = 2
    channels = 2
    length = 5

    enc = Conv1dEncoder(
        input_shape=(channels, length),
        channels=[channels],
        kernels=[1],
        strides=[1],
        padding=[0],
        channels_last=False,
    )

    inp = torch.rand((bsz, channels, length))
    out = enc(inp)

    # test that the shape of the output matches the calculated one
    output_size = enc.get_encoder_output_size()  # length of flattened output
    output_shape = (bsz, output_size)
    assert out.shape == output_shape

    # test that fails with wrong dimensions
    inp_wrong_dim = torch.rand((bsz, length + 2, channels))
    with pytest.raises(
        RuntimeError,
        match=".*to have [0-9]+ channels, but got [0-9]+ channels instead$",
    ):
        _ = enc(inp_wrong_dim)

    # test that input gets permuted when channels_last = True
    enc = Conv1dEncoder(
        input_shape=(length, channels),
        channels=[channels],
        kernels=[1],
        strides=[1],
        padding=[0],
        channels_last=True,
    )
    assert tuple(enc._input_shape) == (channels, length)
