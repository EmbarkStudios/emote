# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This file contains codes/text mostly restructured from the following github repository
# https://github.com/facebookresearch/mbrl-lib

import torch
import numpy as np
from torch import nn, optim
from typing import Tuple, Dict, Optional, List, Any, Union

from emote.utils.math import Normalizer
from emote.utils.model import to_tensor
from emote.typing import TensorType

from emote.callbacks import LossCallback


class DynamicModel(nn.Module):
    """ Wrapper class for model.

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
        """ Computes the output of the dynamics model.
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
        reward: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """ Computes the model loss over a batch of transitions.

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
        model_in, target = self.process_batch(obs=obs, next_obs=next_obs, action=action, reward=reward)
        return self.model.loss(model_in, target=target)

    def update(
        self,
        *,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Updates the model given a batch of transitions and an optimizer.

        Args:
            obs (tensor): current observations
            next_obs (tensor): next observations
            action (tensor): actions
            reward (tensor): rewards
            optimizer (torch optimizer): the optimizer to use to update the model.

        Returns:
            (tensor and optional dict): as returned by `model.loss().`
        """
        model_in, target = self.process_batch(obs=obs, action=action, next_obs=next_obs, reward=reward)
        return self.model.update(model_in, optimizer, target=target)

    def reset(
        self, obs: torch.Tensor, rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """ Initializes the model to start a new simulated trajectory.
            This method can be used to initialize data that should be kept constant during
            a simulated trajectory starting at the given observation (for example model
            indices when using a bootstrapped ensemble with TSinf propagation). It should
            also return any state produced by the model that the :meth:`sample()` method
            will require to continue the simulation (e.g., predicted observation,
            latent state, last action, beliefs, propagation indices, etc.).

            Args:
                obs (tensor): the observation from which the trajectory will be
                    started.
                rng (torch.generator, optional): an optional random number generator

            Returns:
                (dict(str, tensor)): the model state necessary to continue the simulation.
        """
        if not hasattr(self.model, "reset_1d"):
            raise RuntimeError(
                "DynamicModel requires wrapped model to define method reset_1d"
            )
        obs = to_tensor(obs).to(self.device)
        model_state = {"obs": obs}
        model_state.update(self.model.reset_1d(obs, rng=rng))
        return model_state

    def sample(
        self,
        act: torch.Tensor,
        model_state: Dict[str, torch.Tensor],
        deterministic: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """ Samples a simulated transition from the dynamics model.
            This method will be used by :class:`ModelEnv` to simulate a transition of the form.
                o_t+1, r_t+1, d_t+1, s_t = sample(a_t, s_t), where

                - a_t: action taken at time t.
                - s_t: model state at time t (as returned by :meth:`reset()` or :meth:`sample()`.
                - r_t: reward at time t.
                - d_t: terminal indicator at time t.

            If the model doesn't simulate rewards and/or terminal indicators, it can return
            ``None`` for those.

            Args:
                act (tensor): the action at.
                model_state (tensor): the model state st.
                deterministic (bool): if ``True``, the model returns a deterministic
                    "sample" (e.g., the mean prediction). Defaults to ``False``.
                rng (generator): optional random generator

            Returns:
                (tuple): predicted observation, rewards, terminal indicator and model
                    state dictionary. Everything but the observation is optional, and can
                    be returned with value ``None``.
        """
        obs = to_tensor(model_state["obs"]).to(self.device)
        model_in = self.get_model_input(model_state["obs"], act)
        if not hasattr(self.model, "sample_1d"):
            raise RuntimeError(
                "DynamicModel requires wrapped model to define method sample_1d"
            )
        preds, next_model_state = self.model.sample_1d(
            model_in, model_state, rng=rng, deterministic=deterministic
        )
        next_observs = preds[:, :-1] if self.learned_rewards else preds
        if self.target_is_delta:
            tmp_ = next_observs + obs
            for dim in self.no_delta_list:
                tmp_[:, dim] = next_observs[:, dim]
            next_observs = tmp_
        rewards = preds[:, -1:] if self.learned_rewards else None
        next_model_state["obs"] = next_observs
        return next_observs, rewards, None, next_model_state

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
        _as_float: bool = False
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

    def get_output_and_targets(
        self,
        *,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Returns the model output and the target tensors given a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.forward()` on them and returns the value.
        No gradient information will be kept.

        Args:
            obs (tensor): current observations
            next_obs (tensor): next observations
            action (tensor): actions
            reward (tensor): rewards

        Returns:
            (tuple(tensor), tensor): the model outputs and the target for this batch.
        """
        with torch.no_grad():
            model_in, target = self.process_batch(obs=obs, action=action, next_obs=next_obs, reward=reward)
            output = self.model.forward(model_in)
        return output, target

    def eval_score(
        self,
        *,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor
    ) -> Union[torch.Tensor, None]:
        """Evaluates the model score over a batch of transitions.

        This method constructs input and targets from the information in the batch,
        then calls `self.model.eval_score()` on them and returns the value.

        Args:
            obs (tensor): current observations
            next_obs (tensor): next observations
            action (tensor): actions
            reward (tensor): rewards

        Returns:
            (tensor): as returned by model.eval_score() or None if the method is not defined
        """
        if hasattr(self.model, "eval_score"):
            with torch.no_grad():
                model_in, target = self.process_batch(obs=obs, action=action, next_obs=next_obs, reward=reward)
                return self.model.eval_score(model_in, target=target)
        return None

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
        )
        self.model = model

    def loss(self, observation, next_observation, actions, rewards):
        loss, loss_info = self.model.loss(obs=observation['obs'],
                                          next_obs=next_observation['obs'],
                                          action=actions,
                                          reward=rewards)
        return loss

