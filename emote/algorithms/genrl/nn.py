import torch

from torch import Tensor, nn

from emote.nn.action_value_mlp import ActionValueMlp
from emote.nn.gaussian_policy import GaussianMlpPolicy


class GenRLPolicy(GaussianMlpPolicy):
    def __init__(
        self,
        decoder: nn.Module,
        num_obs: int,
        num_actions: int,
        hidden_dims: list[int],
    ):
        super().__init__(num_obs, num_actions, hidden_dims)
        self.num_actions = num_actions
        self.decoder = decoder

    def forward(self, obs: Tensor, epsilon: Tensor = None):
        obs = obs.to(torch.float)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(dim=0)

        if epsilon is not None:
            epsilon = epsilon[:, : self.num_actions]

        if self.training:
            sample, log_prob = super().forward(obs, epsilon)
            action = self.decoder(sample, obs)
            return action, log_prob

        return self.decoder(super().forward(obs, epsilon), obs)


class GenRLQNet(ActionValueMlp):
    def __init__(self, action_encoder, num_obs, num_actions, hidden_dims):
        super().__init__(num_obs, num_actions, hidden_dims)
        self.action_encoder = action_encoder
        self.num_action = num_actions

    def forward(self, action, obs):

        if len(action.shape) == 1:
            action = action.unsqueeze(dim=0)

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(dim=0)

        if action.shape[1] != self.num_action:
            latent_action = self.action_encoder(
                action,
                obs,
            )
            return super().forward(latent_action, obs)
        return super().forward(action, obs)
