from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from emote.nn.initialization import ortho_init_


class Conv2dEncoder(nn.Module):
    """
    Multi-layer 2D convolutional encoder.

    :param input_shape: (tuple[int, int, int]) The input image shape, this should be consistent with channels_last.
    :param channels: (list[int]) The number of channels for each conv layer.
    :param kernels: (list[int]) The kernel size for each conv layer.
    :param strides: (list[int]) The strides for each conv layer.
    :param padding: (list[int]]) The padding.
    :param channels_last: (bool) Whether the input image has channels as the last dim, else first.
    :param activation: (torch.nn.Module) The activation function.
    :param flatten: (bool) Flattens the output into a vector.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        channels: list[int],
        kernels: list[int],
        strides: list[int],
        padding: list[int],
        channels_last: bool = True,
        activation: torch.nn.Module = torch.nn.ReLU,
        flatten: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._channels_last = channels_last
        if channels_last:
            self._img_shape_cwh = [input_shape[2], input_shape[0], input_shape[1]]
        else:
            self._img_shape_cwh = input_shape

        self._channels = channels
        self._kernels = kernels
        self._strides = strides
        self._padding = padding
        self.flatten = flatten

        num_layers = len(channels)
        channels = [self._img_shape_cwh[0]] + channels

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

        if self.flatten:
            self._layers.append(nn.Flatten())

        self.apply(ortho_init_)

    def forward(self, obs: torch.Tensor):
        x = obs
        if self._channels_last:
            x = x.permute(0, 3, 1, 2)
        for layer in self._layers:
            x = layer(x)
        return x

    def get_encoder_output_size(self, return_flattened: bool = False):
        curr_size_x, curr_size_y = self._img_shape_cwh[1], self._img_shape_cwh[2]

        """Calculate the outputs size of a conv encoder."""
        for k, s, p in zip(self._kernels, self._strides, self._padding):
            curr_size_x = ((curr_size_x - k + 2 * p) // s) + 1
            curr_size_y = ((curr_size_y - k + 2 * p) // s) + 1

        out_size = (self._channels[-1], curr_size_x, curr_size_y)

        if self.flatten or return_flattened:
            flattened_out_size = np.prod(out_size)
            return *out_size, flattened_out_size

        return out_size


class Conv1dEncoder(nn.Module):
    """
    Multi-layer 1D convolutional encoder

    :param input_shape: (tuple[int, int]) The input shape
    :param channels: (list[int]) The number of channels for each conv layer.
    :param kernels: (list[int]) The kernel size for each conv layer.
    :param strides: (list[int]) The strides for each conv layer.
    :param padding: (list[int]) The padding.
    :param activation: (torch.nn.Module) The activation function.
    :param flatten: (bool) Flattens the output into a vector.
    :param name: (str) Name of the encoder (default: "conv1d")
    :param channels_last: (bool) Whether the input has channels as the last dim, else first.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        channels: list[int],
        kernels: list[int],
        strides: list[int],
        padding: list[int],
        activation: torch.nn.Module = torch.nn.ReLU,
        flatten: bool = True,
        name: str = "conv1d",
        channels_last: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._channels_last = channels_last
        if channels_last:
            self._input_shape = [input_shape[1], input_shape[0]]
        else:
            self._input_shape = input_shape

        self._channels = channels
        self._kernels = kernels
        self._strides = strides
        self._padding = padding
        self.name = name
        self.flatten = flatten

        num_layers = len(channels)
        channels = [self._input_shape[0]] + channels

        self._layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self._layers.append(
                torch.nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernels[i],
                    strides[i],
                    padding=padding[i],
                )
            )
            self._layers.append(activation())

        if self.flatten:
            self._layers.append(nn.Flatten())

        self.apply(ortho_init_)

    def forward(self, obs: torch.Tensor):
        x = obs
        if self._channels_last:
            x = x.permute(0, 2, 1)
        for layer in self._layers:
            x = layer(x)
        return x

    def get_encoder_output_size(self, return_flattened: bool = False):
        curr_size = self._input_shape[1]

        for k, s, p in zip(self._kernels, self._strides, self._padding):
            curr_size = ((curr_size - k + 2 * p) // s) + 1

        out_size = (self._channels[-1], curr_size)

        if self.flatten or return_flattened:
            flattened_out_size = np.prod(out_size)
            return *out_size, flattened_out_size

        return out_size
