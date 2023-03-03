from functools import partial

import torch

from torch import Tensor, nn

from emote.nn.initialization import ortho_init_


class ActionValueMlp(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims):
        super().__init__()
        self.obs_d = observation_dim
        self.act_d = action_dim
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip(
                    [observation_dim + action_dim] + hidden_dims, hidden_dims
                )
            ],
        )
        self.encoder.apply(ortho_init_)

        self.final_layer = nn.Linear(hidden_dims[-1], 1)
        self.final_layer.apply(partial(ortho_init_, gain=1))

    def forward(self, action: Tensor, obs: Tensor) -> Tensor:
        bsz, obs_d = obs.shape
        bsz_action, act_d = action.shape
        assert bsz == bsz_action
        assert obs_d == self.obs_d
        assert act_d == self.act_d
        x = torch.cat([obs, action], dim=1)
        out = self.final_layer(self.encoder(x))
        assert (bsz, 1) == out.shape
        return out


class SharedEncoderActionValueNet(nn.Module):
    def __init__(
        self,
        shared_enc: nn.Module,
        encoder_out_dim: int,
        action_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        self.shared_enc = shared_enc
        self.action_value_mlp = ActionValueMlp(encoder_out_dim, action_dim, hidden_dims)

    def forward(self, action: torch.Tensor, obs: torch.Tensor):
        x = self.shared_enc(obs)
        value = self.action_value_mlp(action, x)
        return value
