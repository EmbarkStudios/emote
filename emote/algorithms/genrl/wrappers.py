from typing import Callable

import torch

from torch import Tensor, nn

from emote.nn.gaussian_policy import GaussianMlpPolicy


class DecoderWrapper(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        condition_fn: Callable,
        latent_multiplier: float = 3.0,
    ):
        super().__init__()
        self.device = decoder.device
        self._latent_multiplier = latent_multiplier
        self.latent_size = decoder.input_size
        self.output_size = decoder.output_size
        self.condition_size = decoder.condition_size
        self.condition_fn = condition_fn
        self.decoder = decoder

    def forward(
        self, latent: torch.Tensor, observation: torch.Tensor = None
    ) -> torch.Tensor:

        """
        Running decoder

            Arguments:
                latent (torch.Tensor): batch x latent_size
                observation (torch.Tensor): batch x obs_size

            Returns:
                torch.Tensor: the sample (batch x data_size)
        """
        latent = latent * self._latent_multiplier

        latent = latent.to(self.device)
        condition = None
        if observation is not None:
            observation = observation.to(self.device)
            condition = self.condition_fn(observation)

        sample = self.decoder.forward(latent, condition)

        return sample

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        assert new_state_dict != {}
        model_dict.update(new_state_dict)
        super().load_state_dict(model_dict)


class EncoderWrapper(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        condition_fn: Callable,
    ):
        super().__init__()
        self.encoder = encoder
        self.device = encoder.device
        self.action_size = encoder.input_size
        self.latent_size = encoder.output_size
        self.condition_size = encoder.condition_size

        self.condition_fn = condition_fn

    def forward(
        self, action: torch.Tensor, observation: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Running encoder

            Arguments:
                action (torch.Tensor): batch x data_size
                observation (torch.Tensor): batch x obs_size

            Returns:
                torch.Tensor: the mean (batch x data_size)
        """
        action = action.to(self.device)
        condition = None
        if observation is not None:
            observation = observation.to(self.device)
            condition = self.condition_fn(observation)

        mean, _ = self.encoder.forward(action, condition)
        return mean

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        new_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        assert new_state_dict != {}
        model_dict.update(new_state_dict)
        super().load_state_dict(model_dict)


class PolicyWrapper(nn.Module):
    def __init__(
        self,
        decoder: DecoderWrapper,
        policy: GaussianMlpPolicy,
    ):
        super().__init__()
        self.latent_size = decoder.latent_size
        self.decoder = decoder
        self.policy = policy

    def forward(self, obs: Tensor, epsilon: Tensor = None):
        # we need to discard the extra dimensions of epsilon.
        # the input epsilon is given for the original action space
        # however, the policy outputs latent actions.
        if epsilon is not None:
            epsilon = epsilon[:, : self.latent_size]

        if self.training:
            sample, log_prob = self.policy.forward(obs, epsilon)
            action = self.decoder(sample, obs)
            return action, log_prob

        return self.decoder(self.policy.forward(obs, epsilon), obs)
