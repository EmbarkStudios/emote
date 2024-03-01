from typing import Callable

import torch

from torch import Tensor, nn

from emote import Callback
from emote.algorithms.amp import gradient_loss_function
from emote.callbacks.logging import LoggingMixin
from emote.callbacks.loss import LossCallback


class DiscriminatorLoss(LossCallback):
    """
    This loss is used to train a discriminator for
    adversarial training.
    """

    def __init__(
        self,
        discriminator: nn.Module,
        encoder: nn.Module,
        animation_rollout_length: int,
        imitation_state_map_fn: Callable[[Tensor], Tensor],
        policy_state_map_fn: Callable[[Tensor], Tensor],
        grad_loss_weight: float,
        optimizer: torch.optim.Optimizer,
        lr_schedule: torch.optim.lr_scheduler._LRScheduler,
        max_grad_norm: float,
        input_key: str = "features",
        latent_key: str = "latent",
        name: str = "Discriminator",
    ):
        super().__init__(
            lr_schedule=lr_schedule,
            name=name,
            network=discriminator,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            data_group=None,
        )
        self._discriminator = discriminator
        self._encoder = encoder
        self._rollout_length = animation_rollout_length
        self._imitation_state_map_function = imitation_state_map_fn
        self._policy_state_map_function = policy_state_map_fn
        self._grad_loss_weight = grad_loss_weight
        self._obs_key = input_key
        self._latent_key = latent_key

    def loss(self, imitation_batch: dict, policy_batch: dict) -> Tensor:
        """
        Computing the loss
            Arguments:
                imitation_batch (dict): a batch of data from the reference animation.
                    the discriminator is trained to classify data from this batch as
                    positive samples
                policy_batch (dict): a batch of data from the RL buffer. the discriminator
                    is trained to classify data from this batch as negative samples.
            Returns:
                loss (Tensor): the loss tensor
        """
        imitation_batch_size = imitation_batch["batch_size"]
        policy_data_batch_size = policy_batch["batch_size"]

        pos_obs: Tensor = imitation_batch["observation"][self._obs_key]
        pos_next_obs: Tensor = imitation_batch["next_observation"][self._obs_key]
        pos_obs = pos_obs.reshape(imitation_batch_size, self._rollout_length, -1)
        pos_next_obs = pos_next_obs.unsqueeze(dim=1)
        rollouts = torch.cat((pos_obs, pos_next_obs), dim=1)

        with torch.no_grad():
            pos_latent = self._encoder(rollouts.view(imitation_batch_size, -1))
        indices = torch.randint(self._rollout_length, (imitation_batch_size,))
        pos_obs = rollouts[torch.arange(imitation_batch_size), indices, :]
        pos_next_obs = rollouts[torch.arange(imitation_batch_size), indices + 1, :]

        neg_obs: Tensor = policy_batch["observation"][self._obs_key]
        neg_next_obs: Tensor = policy_batch["next_observation"][self._obs_key]
        neg_latent: Tensor = policy_batch["observation"][self._latent_key]

        pos_obs = self._imitation_state_map_function(pos_obs)
        pos_next_obs = self._imitation_state_map_function(pos_next_obs)
        pos_input = torch.cat([pos_obs, pos_next_obs], dim=-1)
        pos_input.requires_grad_(True)

        neg_obs = self._policy_state_map_function(neg_obs)
        neg_next_obs = self._policy_state_map_function(neg_next_obs)
        neg_input = torch.cat([neg_obs, neg_next_obs], dim=-1)

        pos_output = self._discriminator(pos_input, pos_latent)
        neg_output = self._discriminator(neg_input, neg_latent)
        assert pos_output.shape == (imitation_batch_size, 1)
        assert neg_output.shape == (policy_data_batch_size, 1)

        pos_loss = torch.mean(torch.square(pos_output - 1.0))  # Positive samples should label to 1.
        neg_loss = torch.mean(
            torch.square(neg_output + 1.0)
        )  # Negative samples should label to -1.

        grad_penalty_loss = self._grad_loss_weight * gradient_loss_function(pos_output, pos_input)

        loss = pos_loss + neg_loss + grad_penalty_loss

        self.log_scalar("calm/loss/pos_discrimination_loss", pos_loss)
        self.log_scalar("calm/loss/neg_discrimination_loss", neg_loss)
        self.log_scalar("calm/loss/grad_loss", grad_penalty_loss)
        self.log_scalar("calm/loss/total", loss)
        self.log_scalar("calm/predict/positive_samples_mean", torch.mean(pos_output))
        self.log_scalar("calm/predict/positive_samples_std", torch.std(pos_output))
        self.log_scalar("calm/predict/negative_samples_mean", torch.mean(neg_output))
        self.log_scalar("calm/predict/negative_samples_std", torch.std(neg_output))

        return loss


class CALMReward(Callback, LoggingMixin):
    """Adversarial rewarding with CALM"""

    def __init__(
        self,
        discriminator: nn.Module,
        state_map_fn: Callable[[Tensor], Tensor],
        style_reward_weight: float,
        rollout_length: int,
        data_group: str,
        observation_key: str = "features",
        latent_key: str = "latent",
    ):
        super().__init__()
        self._discriminator = discriminator
        self._order = 1
        self.data_group = data_group
        self._style_reward_weight = style_reward_weight
        self._state_map_function = state_map_fn
        self._rollout_length = rollout_length
        self._obs_key = observation_key
        self._latent_key = latent_key

    def begin_batch(
        self, observation: dict[str, Tensor], next_observation: dict[str, Tensor], rewards: Tensor
    ):
        """
        Updating the reward by adding the weighted AMP reward
            Arguments:
                observation: current observation
                next_observation: next observation
                rewards: task reward
            Returns
                dict: the batch data with updated reward
        """
        obs = observation[self._obs_key]
        latent = observation[self._latent_key]
        batch_size = obs.shape[0]
        obs_unrolled = obs.reshape(batch_size // self._rollout_length, self._rollout_length, -1)
        next_obs = next_observation[self._obs_key]
        next_obs = next_obs.unsqueeze(dim=1)

        rollouts = torch.cat((obs_unrolled, next_obs), dim=1)
        next_obs = rollouts[:, 1:]

        next_obs = next_obs.reshape(batch_size, -1)

        state = self._state_map_function(obs)
        next_state = self._state_map_function(next_obs)

        consecutive_states = torch.cat([state, next_state], dim=-1)

        predictions = self._discriminator(consecutive_states, latent).detach()

        style_reward = 1.0 - 0.25 * (predictions - 1.0) ** 2.0
        scaled_style_reward = self._style_reward_weight * style_reward
        assert scaled_style_reward.shape == rewards.shape

        total_reward = rewards + scaled_style_reward

        self.log_scalar("calm/unscaled_style_reward", torch.mean(style_reward))
        self.log_scalar("calm/task_reward", torch.mean(rewards))
        self.log_scalar("calm/scaled_style_reward", torch.mean(scaled_style_reward))
        self.log_scalar("calm/total_reward", torch.mean(total_reward))
        self.log_scalar("calm/predicts_mean", torch.mean(predictions))
        self.log_scalar("calm/predicts_std", torch.std(predictions))

        return {self.data_group: {"rewards": total_reward}}
