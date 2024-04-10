from __future__ import annotations

from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from emote.nn.gaussian_policy import GaussianPolicyHead
from typing import Mapping, Optional, Union
from torch import Tensor, nn
from emote.nn.initialization import ortho_init_
from emote.nn.gaussian_policy import GaussianMlpPolicy


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


class TNet(nn.Module):
    def __init__(
            self,
            num_points: int,
            input_dim: int,
            input_stack_hidden_dims: list[int] = None,
            feature_stack_hidden_dims: list[int] = None,
            use_batch_norm: bool = False,
            device: torch.device = torch.device("cpu"),
    ):
        """
        TNet mapping in PointNet architecture
        :param num_points (int): number of points
        :param input_dim (int): input dimension of points in the point-cloud
        :param input_stack_hidden_dims (list[int]): size of hidden MLP at input stack
        :param feature_stack_hidden_dims (list[int]): size of hidden MLP at feature stack
        :param use_batch_norm (bool): whether to use batch norm in the architecture
        :param device (torch.device): the device to train/deploy the model
        """
        super(TNet, self).__init__()

        # setting the default hidden sizes in case None given
        if input_stack_hidden_dims is None:
            input_stack_hidden_dims = [64, 128, 1024]
        if feature_stack_hidden_dims is None:
            feature_stack_hidden_dims = [512, 256]

        self._input_dim = input_dim
        self._num_points = num_points
        self._device = device
        self._use_batch_norm = use_batch_norm

        # creating the input MLP stack
        input_mlp = [
            nn.Sequential(
                nn.Conv1d(dim_in, dim_out, kernel_size=(1,)),
                nn.ReLU(),
                nn.BatchNorm1d(dim_out) if self._use_batch_norm else nn.Identity(),
            )
            for dim_in, dim_out in zip(
                [self._input_dim] + input_stack_hidden_dims, input_stack_hidden_dims
            )
        ]
        # adding the max pool
        input_mlp.append(nn.Sequential(nn.MaxPool1d(kernel_size=self._num_points)))
        # creating the features MLP stack
        feature_mlp = [
            nn.Sequential(
                nn.Linear(dim_in, dim_out),
                nn.ReLU(),
                nn.BatchNorm1d(dim_out) if self._use_batch_norm else nn.Identity(),
            )
            for dim_in, dim_out in zip(
                [input_stack_hidden_dims[-1]] + feature_stack_hidden_dims, feature_stack_hidden_dims
            )
        ]
        feature_mlp.append(
            nn.Sequential(nn.Linear(feature_stack_hidden_dims[-1], self._input_dim ** 2))
        )
        self.input_mlp = nn.Sequential(*input_mlp).to(self._device)
        self.input_mlp.apply(ortho_init_)
        self.feature_mlp = nn.Sequential(*feature_mlp).to(self._device)
        self.feature_mlp.apply(ortho_init_)

    def forward(self, x):
        # shape of x: [bs, feature_size, num_points]
        bs = x.shape[0]
        x = self.input_mlp(x)
        x = x.squeeze(-1)
        x = self.feature_mlp(x)
        eye = torch.eye(self._input_dim, requires_grad=True).repeat(bs, 1, 1).to(self._device)
        x = x.view(bs, self._input_dim, self._input_dim) + eye
        return x


class PointNetEncoder(nn.Module):
    def __init__(
            self,
            num_points: int,
            input_dim: int,
            output_dim: int,
            input_stack_hidden_dims: list[int] = None,
            feature_stack_hidden_dims: list[int] = None,
            use_tnet: bool = False,
            use_batch_norm: bool = False,
            device: torch.device = torch.device("cpu"),
    ):
        """
        PointNet encoder class
        :param num_points: number of points in the point-cloud
        :param input_dim: input dimension of points in the point-cloud
        :param output_dim: output feature dimension size
        :param input_stack_hidden_dims: hidden sizes for the input MLP stack
        :param feature_stack_hidden_dims: hidden sizes for the feature MLP stack
        :param use_tnet: whether to include TNet in the architecture
        :param use_batch_norm: whether to include batch norm in the architecture
        :param device: device to train/run the model
        """
        super().__init__()

        # setting default hidden sizes in case None given
        if input_stack_hidden_dims is None:
            input_stack_hidden_dims = [64, 64]
        if feature_stack_hidden_dims is None:
            feature_stack_hidden_dims = [64, 128]

        self._num_points = num_points
        self._output_dim = output_dim
        self._input_dim = input_dim
        self._device = device
        self._use_tnet = use_tnet
        self._use_batch_norm = use_batch_norm

        if self._use_tnet:
            # creating the TNet models
            self.t_net1 = TNet(
                num_points=self._num_points,
                input_dim=input_dim,
                use_batch_norm=self._use_batch_norm,
                device=self._device,
            )
            self.t_net2 = TNet(
                num_points=self._num_points,
                input_dim=input_stack_hidden_dims[-1],
                use_batch_norm=self._use_batch_norm,
                device=self._device,
            )

        # creating the input MLP stack
        mlp_stack1 = [
            nn.Sequential(
                nn.Conv1d(dim_in, dim_out, kernel_size=(1,)),
                nn.ReLU(),
                nn.BatchNorm1d(dim_out) if self._use_batch_norm else nn.Identity(),
            )
            for dim_in, dim_out in zip(
                [self._input_dim] + input_stack_hidden_dims, input_stack_hidden_dims
            )
        ]
        # creating the features MLP stack
        mlp_stack2 = [
            nn.Sequential(
                nn.Conv1d(dim_in, dim_out, kernel_size=(1,)),
                nn.ReLU(),
                nn.BatchNorm1d(dim_out) if self._use_batch_norm else nn.Identity(),
            )
            for dim_in, dim_out in zip(
                [input_stack_hidden_dims[-1]] + feature_stack_hidden_dims,
                feature_stack_hidden_dims + [self._output_dim],
            )
        ]
        mlp_stack2.append(nn.Sequential(nn.MaxPool1d(kernel_size=self._num_points)))
        if self._use_tnet:
            self.mlp_stack1 = nn.Sequential(*mlp_stack1).to(self._device)
            self.mlp_stack2 = nn.Sequential(*mlp_stack2).to(self._device)
            self.mlp_stack1.apply(ortho_init_)
            self.mlp_stack2.apply(ortho_init_)
        else:
            self.mlp_model = mlp_stack1 + mlp_stack2
            self.mlp_model = nn.Sequential(*self.mlp_model).to(self._device)
            self.mlp_model.apply(ortho_init_)

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1, self._input_dim)
        if self._use_tnet:
            return self.forward_with_tnet(x)
        else:
            return self.forward_without_tnet(x)

    def forward_with_tnet(self, x):
        # x.shape: [bs, num_points, dim_points]
        x = x.transpose(2, 1)

        # get the transformation matrix 1
        t_mat = self.t_net1(x)
        # apply the transformation matrix
        x = torch.bmm(x.transpose(2, 1), t_mat).transpose(2, 1)
        # pass inputs through the first MLP stack
        x = self.mlp_stack1(x)

        # get the transformation matrix 2
        t_mat = self.t_net2(x)
        # apply the transformation matrix
        x = torch.bmm(x.transpose(2, 1), t_mat).transpose(2, 1)
        # pass inputs through the second MLP stack
        features = self.mlp_stack2(x).squeeze(-1)
        return features

    def forward_without_tnet(self, x):
        # x.shape: [bs, num_points, dim_points]
        features = self.mlp_model(x.transpose(2, 1)).squeeze(-1)
        return features


class SharedEncoderPolicy(GaussianMlpPolicy):
    def __init__(
            self,
            shared_enc: nn.Module,
            observation_dim: int,
            action_dim: int,
            hidden_dims: list[int],
    ):
        """
        Policy class: The policy gets an encoder network that maps input observations (in this
        case, images) into a D-dimensional representation. The policy class adds some extra MLP
        layers, and a Guassian head to implement the final stochastic policy. Extra features can
        be flexibly added as the policy input by the forward method which can be overridden to
        accommodate different input observations (input keys).
        :params shared_enc (nn.Module): The shared encoder network between the policy and the
            q-network. Note that the shared encoder should not be trained by the policy loss
        :observation_dim (int): The observation size (the input dimension of the MLP layers)
        :action_dim (int): The action size (policy output size)
        :hidden_dim (list[int]): The MLP hidden layer dimensions.
        """
        super().__init__(
            observation_dim=observation_dim, action_dim=action_dim, hidden_dims=hidden_dims
        )
        self.shared_enc = shared_enc

    def forward(
            self, images: Tensor, features: Tensor, epsilon: Union[Tensor, None] = None, **kwargs
    ) -> Union[Tensor, tuple[Tensor]]:
        """
        The forward method of the policy. This method can be overridden by the child
        class to accommodate various observation keys.
        """
        enc_im = self.shared_enc(images)
        x = torch.cat([enc_im, features], dim=1)
        return super().forward(x, epsilon=epsilon)

    def get_trainable_params(self):
        # ** This is critical for training! **
        # Prevent the policy loss from training the shared encoder

        # IMPORTANT. When using a shared encoder, this filtering is critical. It may not seem so
        # at first, if we use a stop_gradient after the encoder for the policy.
        # However, because Q and P share an encoder and we have to backpropagate the policy sample
        # though the q network, if this filtering isn't present, the policy loss actually trains the
        # encoder indirectly through the q(obs, sample) evaluation.
        trainable_params = list(self.mlp_encoder.parameters()) + list(self.policy.parameters())

        return trainable_params


class PointCloudPolicy(SharedEncoderPolicy):
    def __init__(
            self,
            shared_enc: nn.Module,
            observation_dim: int,
            action_dim: int,
            hidden_dims: list[int],
    ):
        super().__init__(shared_enc, observation_dim, action_dim, hidden_dims)

    def forward(self, point_cloud: Tensor, features: Tensor, epsilon: Union[Tensor, None] = None):
        return super().forward(point_cloud, features, epsilon)


class QNet(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.obs_d = observation_dim
        self.act_d = action_dim
        self.mlp_encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip(
                    [observation_dim + action_dim] + hidden_dims,
                    hidden_dims,
                )
            ],
        )
        self.mlp_encoder.apply(ortho_init_)

        self.final_layer = nn.Linear(hidden_dims[-1], 1)
        self.final_layer.apply(partial(ortho_init_, gain=1))

    def forward(
        self,
        action: Tensor,
        features: Tensor,
    ) -> Tensor:
        bsz, obs_d = features.shape
        bsz_action, act_d = action.shape
        assert bsz == bsz_action
        assert obs_d == self.obs_d
        assert act_d == self.act_d
        x = torch.cat([features, action], dim=1)
        out = self.final_layer(self.mlp_encoder(x))
        assert (bsz, 1) == out.shape
        return out


class SharedEncoderQNet(QNet):
    def __init__(
        self,
        shared_enc: nn.Module,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__(observation_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims)
        self.shared_enc = shared_enc

    def forward(self, action: Tensor, features: Tensor, images: Tensor, **kwargs):
        x = self.shared_enc(images)
        x = torch.cat([x, features], dim=1)
        return super().forward(action, x)


class PointCloudQNet(SharedEncoderQNet):
    def __init__(
        self,
        shared_enc: nn.Module,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__(shared_enc, obs_dim, action_dim, hidden_dims)

    def forward(
        self,
        action: Tensor,
        features: Tensor,
        point_cloud: Tensor,
    ):
        return super().forward(action, features, point_cloud)