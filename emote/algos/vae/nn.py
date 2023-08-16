from typing import Callable

import torch

from torch import nn

from emote.nn.layers import FullyConnectedDecoder, FullyConnectedEncoder


class DecoderWrapper(nn.Module):
    def __init__(
        self,
        latent_size: int,
        output_size: int,
        condition_fn: Callable,
        condition_size: int,
        device: torch.device,
        hidden_sizes: list[int],
        latent_multiplier: float = 3.0,
    ):
        super().__init__()
        self.device = device
        self._latent_multiplier = latent_multiplier
        self.latent_size = latent_size
        self.output_size = output_size
        self.condition_size = condition_size
        self.condition_fn = condition_fn
        self.decoder = FullyConnectedDecoder(
            latent_size=latent_size,
            output_size=output_size,
            device=device,
            condition_size=condition_size,
            hidden_sizes=hidden_sizes,
        )

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
        assert len(latent.shape) == 2

        latent = latent * self._latent_multiplier

        latent = latent.to(self.device)
        condition = None
        if observation is not None:
            assert len(observation.shape) == 2
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
        action_size: int,
        latent_size: int,
        device: torch.device,
        condition_fn: Callable,
        condition_size: int,
        hidden_sizes: list[int],
    ):
        super().__init__()
        self.device = device

        self.action_size = action_size
        self.latent_size = latent_size

        self.condition_fn = condition_fn
        self.condition_size = condition_size

        self.encoder = FullyConnectedEncoder(
            input_size=action_size,
            output_size=latent_size,
            device=device,
            condition_size=condition_size,
            hidden_sizes=hidden_sizes,
        )

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
        assert len(action.shape) == 2
        action = action.to(self.device)
        condition = None
        if observation is not None:
            assert len(observation.shape) == 2
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
