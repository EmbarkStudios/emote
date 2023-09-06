import torch

from torch import Tensor, nn

from emote.nn.action_value_mlp import ActionValueMlp
from emote.nn.gaussian_policy import GaussianMlpPolicy
from emote.algorithms.vae.nn import EncoderWrapper


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

    def forward(self, features: Tensor, epsilon: Tensor = None):
        features = features.to(torch.float)
        if len(features.shape) == 1:
            features = features.unsqueeze(dim=0)

        if epsilon is not None:
            epsilon = epsilon[:, : self.num_actions]

        if self.training:
            sample, log_prob = super().forward(features, epsilon)
            action = self.decoder(sample, features)
            return action, log_prob

        return self.decoder(super().forward(features, epsilon), features)


class GenRLQNet(ActionValueMlp):
    def __init__(
            self,
            action_encoder: EncoderWrapper,
            num_obs: int,
            num_actions: int,
            hidden_dims: int
    ):
        super().__init__(num_obs, num_actions, hidden_dims)
        self.action_encoder = action_encoder
        self.num_action = num_actions

    def forward(self, action: Tensor, features: Tensor):

        if len(action.shape) == 1:
            action = action.unsqueeze(dim=0)

        if len(features.shape) == 1:
            features = features.unsqueeze(dim=0)

        if action.shape[1] != self.num_action:
            latent_action = self.action_encoder(
                action,
                features,
            )
            return super().forward(latent_action, features)
        return super().forward(action, features)
