import torch
from torch import Tensor
from emote.callbacks.loss import LossCallback
from typing import Callable


def gradient_loss_function(prediction: Tensor, inputs: Tensor) -> Tensor:
    imitation_batch_size = prediction.shape[0]
    n_obs = inputs.shape[1]
    predictions = torch.split(prediction, 1, dim=0)
    grad_inp = torch.autograd.grad(predictions, inputs, create_graph=True, retain_graph=True)
    grad_inp = torch.cat(grad_inp, axis=1)
    assert grad_inp.shape == (imitation_batch_size, n_obs)
    grad_inp_norm = torch.square(grad_inp)
    assert grad_inp_norm.shape == (imitation_batch_size, n_obs)
    grad_inp_norm = torch.sum(grad_inp_norm, dim=1)
    assert grad_inp_norm.shape == (imitation_batch_size,)
    grad_inp_loss = torch.mean(grad_inp_norm)
    assert grad_inp_loss.shape == ()
    return grad_inp_loss


class DiscriminatorLoss(LossCallback):
    def __init__(
        self,
        discriminator: Discriminator,
        state_map_fn: Callable[[Tensor], Tensor],
        grad_loss_weight: float,
        optimizer: torch.optim.Optimizer,
        lr_schedule: torch.optim.lr_scheduler._LRScheduler,
        max_grad_norm: float,
        input_key: str = "features",
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
        self._state_map_function = state_map_fn
        self._grad_loss_weight = grad_loss_weight
        self._obs_key = input_key

    def loss(self, imitation_batch: dict, policy_batch: dict) -> Tensor:
        imitation_batch_size = imitation_batch["batch_size"]
        policy_data_batch_size = policy_batch["batch_size"]

        pos_obs: Tensor = imitation_batch["observation"][self._obs_key]
        pos_next_obs: Tensor = imitation_batch["next_observation"][self._obs_key]
        neg_obs: Tensor = policy_batch["observation"][self._obs_key]
        neg_next_obs: Tensor = policy_batch["next_observation"][self._obs_key]

        (pos_obs,) = (self._state_map_function(pos_obs),)
        pos_next_obs = self._state_map_function(pos_next_obs)
        pos_input = torch.cat([pos_obs, pos_next_obs], dim=-1)
        pos_input.requires_grad_(True)

        neg_obs = self._state_map_function(neg_obs)
        neg_next_obs = self._state_map_function(neg_next_obs)
        neg_input = torch.cat([neg_obs, neg_next_obs], dim=-1)

        pos_output = self._discriminator(pos_input)
        neg_output = self._discriminator(neg_input)
        assert pos_output.shape == (imitation_batch_size, 1)
        assert neg_output.shape == (policy_data_batch_size, 1)

        pos_loss = torch.mean(torch.square(pos_output - 1.0))  # Positive samples should label to 1.
        neg_loss = torch.mean(
            torch.square(neg_output + 1.0)
        )  # Negative samples should label to -1.

        grad_penalty_loss = self._grad_loss_weight * gradient_loss_function(pos_output, pos_input)

        loss = pos_loss + neg_loss + grad_penalty_loss

        self.log_scalar("amp/loss/pos_discrimination_loss", pos_loss)
        self.log_scalar("amp/loss/neg_discrimination_loss", neg_loss)
        self.log_scalar("amp/loss/grad_loss", grad_penalty_loss)
        self.log_scalar("amp/loss/total", loss)
        self.log_scalar("amp/predict/positive_samples_mean", torch.mean(pos_output))
        self.log_scalar("amp/predict/positive_samples_std", torch.std(pos_output))
        self.log_scalar("amp/predict/negative_samples_mean", torch.mean(neg_output))
        self.log_scalar("amp/predict/negative_samples_std", torch.std(neg_output))

        return loss


class AMPReward(Callback, LoggingMixin):
    def __init__(
        self,
        discriminator: Discriminator,
        state_map_fn: Callable[[Tensor], Tensor],
        style_reward_weight: float,
        rollout_length: int,
        observation_key: str,
        data_group: str,
    ):
        super().__init__()
        self._discriminator = discriminator
        self._order = 0
        self.data_group = data_group
        self._style_reward_weight = style_reward_weight
        self._state_map_function = state_map_fn
        self._rollout_length = rollout_length
        self._obs_key = observation_key

    def begin_batch(
        self, observation: dict[str, Tensor], next_observation: dict[str, Tensor], rewards: Tensor
    ):
        obs = observation[self._obs_key]
        bsz = obs.shape[0]
        rollouts = obs.reshape(bsz // self._rollout_length, self._rollout_length, -1)
        next_obs = next_observation[self._obs_key]

        next_obs = next_obs.unsqueeze(dim=1)
        combined_obs = torch.cat((rollouts, next_obs), dim=1)
        next_obs = combined_obs[:, 1:]

        next_obs = next_obs.reshape(bsz, -1)

        state = self._state_map_function(obs)
        next_state = self._state_map_function(next_obs)

        consecutive_states = torch.cat([state, next_state], dim=-1)

        predictions = self._discriminator(consecutive_states).detach()

        style_reward = 1.0 - 0.25 * (predictions - 1.0) ** 2.0
        scaled_style_reward = self._style_reward_weight * style_reward
        assert scaled_style_reward.shape == rewards.shape

        total_reward = rewards + scaled_style_reward

        self.log_scalar("amp/unscaled_style_reward", torch.mean(style_reward))
        self.log_scalar("amp/task_reward", torch.mean(rewards))
        self.log_scalar("amp/scaled_style_reward", torch.mean(scaled_style_reward))
        self.log_scalar("amp/total_reward", torch.mean(total_reward))
        self.log_scalar("amp/predicts_mean", torch.mean(predictions))
        self.log_scalar("amp/predicts_std", torch.std(predictions))

        return {self.data_group: {"rewards": total_reward}}