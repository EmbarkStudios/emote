from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from emote.nn.initialization import ortho_init_


class Conv2dEncoder(nn.Module):
    """
    Multi-layer 2D convolutional encoder.

    :param input_channels: (int) The number of input channels
    :param params: (Conv2dParams) The parameters for the conv2d layers
    """

    def __init__(
        self,
        input_shape: Tuple[
            int, int, int
        ],  # (Tuple(int, int, int)) The input image shape (w, h, c).
        channels: List[int],  # (List[int]) The number of channels for each conv layer.
        kernels: List[int],  # (List[int]) The kernel size for each conv layer.
        strides: List[int],  # (List[int]) The strides for each conv layer.
        padding: List[int],  # (List[in]) The padding.
        permute_obs_channels: bool = True,
        activation: torch.nn.Module = torch.nn.ReLU,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_shape = input_shape
        self._channels = channels
        self._kernels = kernels
        self._strides = strides
        self._padding = padding
        self._permute_obs_channels = permute_obs_channels

        num_layers = len(channels)
        channels = [input_shape[2]] + channels

        self._layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self._layers.append(
                torch.nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernels[i],
                    stride=strides[i],
                    padding=padding[i],
                )
            )
            self._layers.append(activation())

        self.apply(ortho_init_)

    def forward(self, obs: torch.Tensor):
        x = obs
        if self._permute_obs_channels:
            x = x.permute(0, 3, 1, 2)
        for layer in self._layers:
            x = layer(x)
        return x

    def get_encoder_output_size(self, flatten: bool = False):
        curr_size_x, curr_size_y = self._input_shape[0], self._input_shape[1]

        """Calculate the outputs size of a conv encoder."""
        for k, s, p in zip(self._kernels, self._strides, self._padding):
            curr_size_x = ((curr_size_x - k + 2 * p) // s) + 1
            curr_size_y = ((curr_size_y - k + 2 * p) // s) + 1

        out_size = (self._channels[-1], curr_size_x, curr_size_y)
        if flatten:
            out_size = np.prod(out_size)
        return out_size
