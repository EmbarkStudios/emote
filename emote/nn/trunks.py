import math

import torch.nn as nn

from torch import Tensor

from emote.nn.initialization import ortho_init_


class Trunk(nn.Module):
    def output_dim(self) -> int:
        ...


class PassThroughTrunk(nn.Module):
    def __init__(
        self,
        tensor_dim: int,
    ):
        super().__init__()
        self.dim = tensor_dim

    def forward(self, x: Tensor) -> Tensor:
        return x

    def output_dim(self) -> int:
        return self.dim


class MlpTrunk(nn.Sequential):
    def __init__(self, observation_dim: int, hidden_dims: list[int]):
        assert len(hidden_dims) > 0
        layers = [
            nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
            for n_in, n_out in zip([observation_dim] + hidden_dims, hidden_dims)
        ]
        super().__init__(*layers)
        self.apply(ortho_init_)
        self.out_dim = hidden_dims[-1]

    def output_dim(self) -> int:
        return self.out_dim


class ResidualBlock(nn.Module):
    """A residual block that is used to create a residual feed-forward trunk.

    It projects the input from the embedding dimension to the hidden dimension,
    applies the ReLU activation, and then projects back to the embedding dimension.
    The output of the residual block is then added to the input.

    Args:
        embedding_dim: The dimension of the embedding.
        hidden_dim: The dimension of the hidden layer.
        use_layer_norm: Whether to use layer normalization.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, use_layer_norm: bool):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x_fc1 = self.fc1(x)
        x_relu = nn.functional.relu(x_fc1)
        return self.fc2(x_relu) + x


class ResidualTrunk(Trunk):
    def __init__(
        self,
        observation_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_layer_norm: bool,
        post_layer_norm: bool,
    ):
        super().__init__()
        fc_obs = nn.Linear(observation_dim, embedding_dim, bias=False)
        self.layers = nn.Sequential(
            fc_obs,
            *[ResidualBlock(embedding_dim, hidden_dim, use_layer_norm) for _ in range(num_layers)],
        )
        self.post_layer_norm = post_layer_norm
        if post_layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_dim)
        self.out_dim = embedding_dim
        ortho_init_(self)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        if self.post_layer_norm:
            x = self.layer_norm(x)
        return x

    def output_dim(self) -> int:
        return self.out_dim


def init_swiglu(m: nn.Module, num_layers: int | None = None):
    if isinstance(m, SwiGLU):
        # fc1: SiLU path → use He/Kaiming (relu is a good proxy for SiLU)
        nn.init.kaiming_uniform_(m.fc1.weight, nonlinearity="relu")

        # fc2: linear gate → Xavier
        nn.init.xavier_uniform_(m.fc2.weight, gain=1.0)

        # fc3: residual projection → either zero or scaled by depth
        if num_layers is None:
            nn.init.zeros_(m.fc3.weight)
        else:
            nn.init.xavier_uniform_(m.fc3.weight, gain=1.0)
            m.fc3.weight.data.mul_(1.0 / math.sqrt(2.0 * num_layers))


class SwiGLU(nn.Module):
    """A SwiGLU layer that is used to create a SwiGLU trunk.

    This layer implements the SwiGLU (Swish-Gated Linear Unit) activation function.
    It normalizes the input if use_layer_norm is True, then projects the input from
    the embedding dimension to the hidden dimension twice. It applies the swish
    activation to one projection and uses the other as a gate, then projects back
    to the embedding dimension. The output is added to the input (residual connection).

    Args:
        embedding_dim: The dimension of the embedding.
        hidden_dim: The dimension of the hidden layer.
        use_layer_norm: Whether to use layer normalization.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, use_layer_norm: bool):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_layer_norm:
            x = self.norm(x)
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x_silu = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x_silu) + x


class SwiGLUTrunk(Trunk):
    def __init__(
        self,
        observation_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_layer_norm: bool,
        post_layer_norm: bool,
    ):
        super().__init__()
        fc_obs = nn.Linear(observation_dim, embedding_dim, bias=False)
        self.layers = nn.Sequential(
            fc_obs,
            *[SwiGLU(embedding_dim, hidden_dim, use_layer_norm) for _ in range(num_layers)],
        )
        self.out_dim = embedding_dim
        init_swiglu(self)
        self.post_layer_norm = post_layer_norm
        if post_layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        if self.post_layer_norm:
            x = self.layer_norm(x)
        return x

    def output_dim(self) -> int:
        return self.out_dim
