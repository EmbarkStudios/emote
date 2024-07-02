from typing import Callable

import torch

from erupt_shared.utils.nn import Discriminator
from torch import Tensor

from emote import Callback
from emote.algorithms.amp import gradient_loss_function
from emote.callbacks.logging import LoggingMixin
from emote.callbacks.loss import LossCallback


def _get_action_mapping_fn(action_indices):
    """Making a mapping function to map an action vector into a lower
    dimensional action vector containing only given action_indices.

    Returns:
        mapping fn: The mapping function
    """

    def action_map_fn(obs: Tensor):
        slices = [obs[:, f.start : f.end] for f in action_indices]
        return torch.cat(slices, dim=1)

    return action_map_fn


class ActionSymmetryDiscriminatorLoss(LossCallback):
    """This loss is used to train a discriminator for adversarial training."""

    def __init__(
        self,
        discriminator: Discriminator,
        right_action_map_fn: Callable[[Tensor], Tensor],
        left_action_map_fn: Callable[[Tensor], Tensor],
        grad_loss_weight: float,
        optimizer: torch.optim.Optimizer,
        lr_schedule: torch.optim.lr_scheduler._LRScheduler,
        max_grad_norm: float,
        data_group: str,
        name: str,
    ):
        super().__init__(
            lr_schedule=lr_schedule,
            name=name,
            network=discriminator,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
        )
        self._discriminator = discriminator
        self._right_action_map_fn = right_action_map_fn
        self._left_action_map_fn = left_action_map_fn
        self._grad_loss_weight = grad_loss_weight

    def loss(self, actions) -> Tensor:
        """Computing the loss to train a discriminator to classify right-side
        from left-side action values."""
        right_actions = self._right_action_map_fn(actions)
        right_actions.requires_grad_(True)

        left_actions = self._left_action_map_fn(actions)

        pos_output = self._discriminator(right_actions)
        neg_output = self._discriminator(left_actions)

        pos_loss = torch.mean(torch.square(pos_output - 1.0))  # Positive samples should label to 1.
        neg_loss = torch.mean(
            torch.square(neg_output + 1.0)
        )  # Negative samples should label to -1.

        grad_penalty_loss = self._grad_loss_weight * gradient_loss_function(
            pos_output, right_actions
        )

        loss = pos_loss + neg_loss + grad_penalty_loss

        self.log_scalar("amp_action_symmetry/loss/pos_discrimination_loss", pos_loss)
        self.log_scalar("amp_action_symmetry/loss/neg_discrimination_loss", neg_loss)
        self.log_scalar("amp_action_symmetry/loss/grad_loss", grad_penalty_loss)
        self.log_scalar("amp_action_symmetry/loss/total", loss)
        self.log_scalar("amp_action_symmetry/predict/positive_samples_mean", torch.mean(pos_output))
        self.log_scalar("amp_action_symmetry/predict/positive_samples_std", torch.std(pos_output))
        self.log_scalar("amp_action_symmetry/predict/negative_samples_mean", torch.mean(neg_output))
        self.log_scalar("amp_action_symmetry/predict/negative_samples_std", torch.std(neg_output))

        return loss


class ActionSymmetryAMPReward(LoggingMixin, Callback):
    """Adversarial rewarding with AMP."""

    def __init__(
        self,
        discriminator: Discriminator,
        right_action_map_fn: Callable[[Tensor], Tensor],
        left_action_map_fn: Callable[[Tensor], Tensor],
        confusion_reward_weight: float,
        data_group: str,
    ):
        super().__init__()
        self._discriminator = discriminator
        self._order = 0
        self.data_group = data_group
        self._reward_weight = confusion_reward_weight
        self._left_action_map_fn = left_action_map_fn
        self._right_action_map_fn = right_action_map_fn

    def begin_batch(self, actions: Tensor, rewards: Tensor):
        """
        Updating the reward by adding the weighted AMP reward
            Arguments:
                actions: batch of actions
                rewards: task reward
            Returns
                dict: the batch data with updated reward
        """
        batch_size = actions.shape[0]

        right_actions = self._right_action_map_fn(actions)
        left_actions = self._left_action_map_fn(actions)

        all_actions = torch.cat((right_actions, left_actions), dim=0)

        predictions = self._discriminator(all_actions)
        predictions_right = predictions[:batch_size]  # labeled 1
        predictions_left = predictions[batch_size:]  # labeled -1

        predictions_right = torch.clamp(predictions_right, -1.0, 1.0)
        predictions_left = torch.clamp(predictions_left, -1.0, 1.0)

        confusion_reward = 1.0 - 0.25 * (predictions_left - 1.0) ** 2
        confusion_reward += 1.0 - 0.25 * (predictions_right + 1.0) ** 2

        scaled_confusion_reward = confusion_reward * self._reward_weight

        total_reward = rewards + scaled_confusion_reward

        self.log_scalar(
            "amp_action_symmetry/unscaled_confusion_reward", torch.mean(confusion_reward)
        )
        self.log_scalar(
            "amp_action_symmetry/scaled_confusion_reward", torch.mean(scaled_confusion_reward)
        )
        self.log_scalar("amp_action_symmetry/task_reward", torch.mean(rewards))
        self.log_scalar("amp_action_symmetry/total_reward", torch.mean(total_reward))

        return {self.data_group: {"rewards": total_reward}}
