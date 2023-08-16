from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from emote.nn.initialization import ortho_init_
from emote.nn.misc import FrozenBatchNorm1d


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

    def get_encoder_output_size(self):
        curr_size_x, curr_size_y = self._img_shape_cwh[1], self._img_shape_cwh[2]

        """Calculate the outputs size of a conv encoder."""
        for k, s, p in zip(self._kernels, self._strides, self._padding):
            curr_size_x = ((curr_size_x - k + 2 * p) // s) + 1
            curr_size_y = ((curr_size_y - k + 2 * p) // s) + 1

        out_size = (self._channels[-1], curr_size_x, curr_size_y)

        if self.flatten:
            out_size = np.prod(out_size)

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

    def get_encoder_output_size(self):
        curr_size = self._input_shape[1]

        for k, s, p in zip(self._kernels, self._strides, self._padding):
            curr_size = ((curr_size - k + 2 * p) // s) + 1

        out_size = (self._channels[-1], curr_size)

        if self.flatten:
            out_size = np.prod(out_size)

        return out_size


class FullyConnectedEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        device: torch.device,
        condition_size: int = 0,
        hidden_sizes: list[int] = None,
        freeze_bn: bool = True,
    ):
        super().__init__()
        self.device = device

        self.input_size = input_size
        self.output_size = output_size
        self._freeze_bn = freeze_bn

        num_layers = len(hidden_sizes)
        batch_norm = (
            FrozenBatchNorm1d(hidden_sizes[0])
            if self._freeze_bn
            else nn.BatchNorm1d(hidden_sizes[0])
        )
        layers = [
            nn.Sequential(
                nn.Linear(input_size + condition_size, hidden_sizes[0]),
                batch_norm,
                nn.ReLU(),
            )
        ]
        for i in range(num_layers - 1):
            batch_norm = (
                FrozenBatchNorm1d(hidden_sizes[i + 1])
                if self._freeze_bn
                else nn.BatchNorm1d(hidden_sizes[i + 1])
            )
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    batch_norm,
                    nn.ReLU(),
                )
            )
        self.hidden_layers = nn.Sequential(*layers).to(self.device)
        self.output_mean = nn.Linear(hidden_sizes[num_layers - 1], output_size).to(
            self.device
        )
        self.output_log_std = nn.Linear(hidden_sizes[num_layers - 1], output_size).to(
            self.device
        )

    def forward(
        self, data: torch.Tensor, condition: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Running encoder

            Arguments:
                data (torch.Tensor): batch x data_size
                condition (torch.Tensor): batch x condition_size

            Returns:
                tuple(torch.Tensor, torch.Tensor): the mean (batch x data_size)
                    and log_std (batch x data_size)
        """
        assert len(data.shape) == 2
        if condition is not None:
            assert len(condition.shape) == 2
            data = torch.cat((data, condition), dim=1)

        x = self.hidden_layers(data)

        mean = self.output_mean(x)
        log_std = self.output_log_std(x)
        return mean, log_std


class FullyConnectedDecoder(nn.Module):
    def __init__(
        self,
        latent_size: int,
        output_size: int,
        device: torch.device,
        condition_size: int = 0,
        hidden_sizes: list[int] = None,
        freeze_bn: bool = True,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 256, 512]

        self.device = device

        self.input_size = latent_size
        self.output_size = output_size
        self._freeze_bn = freeze_bn

        num_layers = len(hidden_sizes)
        batch_norm = (
            FrozenBatchNorm1d(hidden_sizes[0])
            if self._freeze_bn
            else nn.BatchNorm1d(hidden_sizes[0])
        )
        layers = [
            nn.Sequential(
                nn.Linear(latent_size + condition_size, hidden_sizes[0]),
                batch_norm,
                nn.ReLU(),
            )
        ]
        for i in range(num_layers - 1):
            batch_norm = (
                FrozenBatchNorm1d(hidden_sizes[i + 1])
                if self._freeze_bn
                else nn.BatchNorm1d(hidden_sizes[i + 1])
            )
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    batch_norm,
                    nn.ReLU(),
                )
            )
        layers.append(
            nn.Sequential(
                nn.Linear(hidden_sizes[num_layers - 1], output_size),
            )
        )
        self.layers = nn.Sequential(*layers).to(self.device)

    def forward(
        self, latent: torch.Tensor, condition: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Running decoder

            Arguments:
                latent (torch.Tensor): batch x latent_size
                condition (torch.Tensor): batch x condition_size

            Returns:
                torch.Tensor: the sample (batch x data_size)
        """
        assert len(latent.shape) == 2

        if condition is not None:

            assert len(condition.shape) == 2

            latent = torch.cat((latent, condition), dim=1)

        x = self.layers(latent)

        return x
