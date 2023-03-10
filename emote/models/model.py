# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file contains codes/text mostly restructured from the following github repository
# https://github.com/facebookresearch/mbrl-lib

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from torch import nn, optim

from emote.callbacks import LossCallback
from emote.typing import TensorType
from emote.utils.math import Normalizer
from emote.utils.model import to_tensor


class DynamicModel(nn.Module):
    """Wrapper class for model.

    DynamicModel class functions as a wrapper for models including ensembles. It also provides
    data manipulations that are common when using dynamics models with observations
    and actions, so that users don't have to manipulate the underlying model's
    inputs and outputs directly (e.g., predicting delta observations, input
    normalization).

    The wrapper assumes that the wrapped model inputs/outputs will be consistent with

        [pred_obs_{t+1}, pred_rewards_{t+1} (optional)] = model([obs_t, action_t]).

    Args:
        model: the model to wrap.
        target_is_delta (bool): if ``True``, the predicted observations will represent
            the difference respect to the input observations.
            That is, ignoring rewards, pred_obs_{t + 1} = obs_t + model([obs_t, act_t]).
            Defaults to ``True``. Can be deactivated per dimension using ``no_delta_list``.
        normalize (bool): if true, the wrapper will create a normalizer for model inputs,
            which will be used every time the model is called using the methods in this
            class. Assumes the given base model has an attributed ``in_size``.
            To update the normalizer statistics, the user needs to call
            :meth:`update_normalizer` before using the model. Defaults to ``False``.
        normalize_double_precision (bool): if ``True``, the normalizer will work with
            double precision.
        learned_rewards (bool): if ``True``, the wrapper considers the last output of the model
            to correspond to reward predictions, and will use it to construct training
            targets for the model and when returning model predictions. Defaults to ``True``.
        obs_process_fn (callable, optional): if provided, observations will be passed through
            this function before being given to the model (and before the normalizer also).
            The processed observations should have the same dimensions as the original.
            Defaults to ``None``.
        no_delta_list (list(int), optional): if provided, represents a list of dimensions over
            which the model predicts the actual observation and not just a delta.
    """

    def __init__(
        self,
        model: nn.Module,
        target_is_delta: bool = True,
        normalize: bool = False,
        normalize_double_precision: bool = False,
        learned_rewards: bool = True,
        obs_process_fn: Optional[nn.Module] = None,
        no_delta_list: Optional[List[int]] = None,
    ):
        super(DynamicModel, self).__init__()
        self.model = model
        self.input_normalizer: Optional[Normalizer] = None
        if normalize:
            self.input_normalizer = Normalizer(
                self.model.in_size,
                self.model.device,
                dtype=torch.double if normalize_double_precision else torch.float,
            )
        self.device = self.model.device
        self.learned_rewards = learned_rewards
        self.target_is_delta = target_is_delta
        self.no_delta_list = no_delta_list if no_delta_list else []
        self.obs_process_fn = obs_process_fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """Computes the output of the dynamics model.
        Args:
            x (tensor): input

        Returns:
            (tuple of tensors): predicted tensors
        """
        return self.model.forward(x, *args, **kwargs)

    def loss(
        self,
        *,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the model loss over a batch of transitions.

        This method constructs input and targets for the model,
        then calls `self.model.loss()` on them and returns the value and the metadata
        as returned by the model.

        Args:
            obs (tensor): current observations
            next_obs (tensor): next observations
            action (tensor): actions
            reward (tensor): rewards

        Returns:
            (tensor and optional dict): as returned by `model.loss().`
        """
        model_in, target = self.process_batch(
            obs=obs, next_obs=next_obs, action=action, reward=reward
        )
        return self.model.loss(model_in, target=target)

    def sample(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Samples a simulated transition from the dynamics model.

        Args:
            action (tensor): the action at.
            observation (tensor): the observation/state st.
            rng (generator): optional random generator

        Returns:
            (tuple): predicted observation and rewards.
        """
        obs = to_tensor(observation).to(self.device)

        model_in = self.get_model_input(obs, action)
        if not hasattr(self.model, "sample"):
            raise RuntimeError(
                "DynamicModel requires wrapped model to implement sample method"
            )
        preds = self.model.sample(model_in, rng=rng)
        if len(preds.shape) != 2:
            raise RuntimeError(
                "Prediction shape is: {} "
                "Predictions must be 'batch_size x length_of_prediction'."
                "Have you forgotten to run propagation on the ensemble?".format(
                    preds.shape
                )
            )
        next_observs = preds[:, :-1] if self.learned_rewards else preds
        if self.target_is_delta:
            tmp_ = next_observs + obs
            for dim in self.no_delta_list:
                tmp_[:, dim] = next_observs[:, dim]
            next_observs = tmp_
        rewards = preds[:, -1:] if self.learned_rewards else None

        return next_observs, rewards

    def get_model_input(
        self,
        obs: TensorType,
        action: TensorType,
    ) -> torch.Tensor:
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        obs = to_tensor(obs).to(self.device)
        action = to_tensor(action).to(self.device)
        model_in = torch.cat([obs, action], dim=obs.ndim - 1)
        if self.input_normalizer:
            # Normalizer lives on device
            model_in = self.input_normalizer.normalize(model_in).float().to(self.device)
        return model_in

    def process_batch(
        self,
        *,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        _as_float: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.target_is_delta:
            target_obs = next_obs - obs
            for dim in self.no_delta_list:
                target_obs[..., dim] = next_obs[..., dim]
        else:
            target_obs = next_obs
        target_obs = to_tensor(target_obs).to(self.device)
        model_in = self.get_model_input(obs, action)
        if self.learned_rewards:
            reward = to_tensor(reward).to(self.device)
            target = torch.cat([target_obs, reward], dim=obs.ndim - 1)
        else:
            target = target_obs
        return model_in.float(), target.float()

    def update_normalizer(self, *, obs, action):
        """Updates the normalizer statistics given a batch of data

        The normalizer will compute mean and standard deviation the obs and action in
        the transition. If an observation processing function has been provided, it will
        be called on ``obs`` before updating the normalizer. Please note that only obs,
        and action are normalized since they are the inputs to the model.

        Args:
            obs (tensor): Observations
            action (tensor): Actions
        """
        if self.input_normalizer is None:
            return
        if obs.ndim == 1:
            obs = obs[None, :]
            action = action[None, :]
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in_np = np.concatenate([obs, action], axis=obs.ndim - 1)
        self.input_normalizer.update_stats(model_in_np)

    def save(self, save_dir: str) -> None:
        self.model.save(save_dir)
        if self.input_normalizer:
            self.input_normalizer.save(save_dir)

    def load(self, load_dir: str) -> None:
        self.model.load(load_dir)
        if self.input_normalizer:
            self.input_normalizer.load(load_dir)


class ModelLoss(LossCallback):
    """Trains a dynamic model by minimizing the model loss

    :param dynamic_model (torch.nn.Module): A deep neural net that predicts next observation and reward.
    :param ln_alpha (torch.tensor): The current weight for the entropy part of the
        soft Q.
    :param q (torch.nn.Module): A deep neural net that outputs the discounted loss
        given the current observations and a given action.
    :param  lr_schedule (torch.optim.lr_scheduler._LRSchedule): Learning rate schedule
        for the optimizer of model.
    :param opt (torch.optim.Optimizer): An optimizer for pi.
    :param q2 (torch.nn.Module): A second deep neural net that outputs the discounted
        loss given the current observations and a given action. This is not necessary
        since it is fine if the policy isn't pessimistic, but can be nice for symmetry
        with the Q-loss.
    :param max_grad_norm (float): Clip the norm of the gradient during backprop using this value.
    :param name (str): The name of the module. Used e.g. while logging.
    :param data_group (str): The name of the data group from which this Loss takes its data.
    """

    def __init__(
        self,
        *,
        model: DynamicModel,
        opt: optim.Optimizer,
        lr_schedule: Optional[optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 10.0,
        name: str = "dynamic_model",
        data_group: str = "default",
    ):
        super().__init__(
            name=name,
            optimizer=opt,
            lr_schedule=lr_schedule,
            network=model,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
            data_group_locked=True,
        )
        self.model = model

    def loss(self, observation, next_observation, actions, rewards):
        loss, loss_info = self.model.loss(
            obs=observation["obs"],
            next_obs=next_observation["obs"],
            action=actions,
            reward=rewards,
        )
        return loss
