import torch
from torch import nn
from torch import Tensor

from emote.nn.initialization import ortho_init_


class ActionValue(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dims):
        super().__init__()
        self.obs_d = observation_dim
        self.act_d = action_dim
        self.seq = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip(
                    [observation_dim + action_dim] + hidden_dims, hidden_dims
                )
            ],
            nn.Linear(hidden_dims[-1], 1)
        )
        self.seq.apply(ortho_init_)

    def forward(self, action: Tensor, obs: Tensor) -> Tensor:
        bsz, obs_d = obs.shape
        bsz_action, act_d = action.shape
        assert bsz == bsz_action
        assert obs_d == self.obs_d
        assert act_d == self.act_d
        x = torch.cat([obs, action], dim=1)
        out = self.seq(x)
        assert (bsz, 1) == out.shape
        return out
