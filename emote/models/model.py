# This file contains codes and texts that are copied from
# https://github.com/facebookresearch/mbrl-lib

from typing import Optional

import torch
import torch.nn.functional as F

from torch import nn

from emote.utils.model import normal_init


class DynamicModel(nn.Module):
    """Wrapper class for model.
    DynamicModel class functions as a wrapper for models including ensembles. It also provides
    data manipulations that are common when using dynamics models with observations
    and actions (e.g., predicting delta observations, input normalization).

    Arguments:
        model: the model to wrap.
        learned_rewards (bool): if True, the wrapper considers the last output of the model
            to correspond to reward predictions.
        obs_process_fn (callable, optional): if provided, observations will be passed through
            this function before being given to the model.
        no_delta_list (list(int), optional): if provided, represents a list of dimensions over
            which the model predicts the actual observation and not just a delta.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        learned_rewards: bool = True,
        obs_process_fn: Optional[nn.Module] = None,
        no_delta_list: Optional[list[int]] = None,
    ):
        super().__init__()
        self.model = model
        self.device = self.model.device
        self.learned_rewards = learned_rewards
        self.no_delta_list = no_delta_list if no_delta_list else []
        self.obs_process_fn = obs_process_fn
        self.input_normalizer = Normalizer()
        self.target_normalizer = Normalizer()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Computes the output of the dynamics model.

        Args:
            x (tensor): input

        Returns:
            (tuple of tensors): predicted tensors
        """
        return self.model.forward(x)

    def loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, any]]:
        """Computes the model loss over a batch of transitions.

        Arguments:
            obs (tensor): current observations
            next_obs (tensor): next observations
            action (tensor): actions
            reward (tensor): rewards

        Returns:
            (tensor and optional dict): the loss tensor and optional info
        """
        model_in, target = self.process_batch(
            obs=obs, next_obs=next_obs, action=action, reward=reward
        )
        return self.model.loss(model_in, target=target)

    def sample(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        rng: torch.Generator,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Samples a simulated transition from the dynamics model. The function first
        normalizes the inputs to the model, and then denormalize the model output as the
        final output.

        Arguments:
            action (tensor): the action at.
            observation (tensor): the observation/state st.
            rng (torch.Generator): a random number generator.

        Returns:
            (tuple): predicted observation and rewards.
        """
        model_in = self.get_model_input(observation, action)

        model_in = self.input_normalizer.normalize(model_in)
        preds = self.model.sample(model_in, rng)
        preds = self.target_normalizer.denormalize(preds)

        assert len(preds.shape) == 2, (
            f"Prediction shape is: {preds.shape} Predictions must be 'batch_size x "
            f"length_of_prediction. Have you forgotten to run propagation on the ensemble?"
        )
        next_observs = preds[:, :-1] if self.learned_rewards else preds

        tmp_ = next_observs + observation
        for dim in self.no_delta_list:
            tmp_[:, dim] = next_observs[:, dim]
        next_observs = tmp_
        rewards = preds[:, -1:] if self.learned_rewards else None

        return next_observs, rewards

    def get_model_input(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """The function prepares the input to the neural network model by concatenating
        observations and actions. In case, obs_process_fn is given, the observations are
        processed by the function prior to the concatenation.

            Arguments:
                 obs (torch.Tensor): observation tensor
                 action (torch.Tensor): action tensor

            Returns:
                (torch.Tensor): the concatenation of obs and actions
        """
        if self.obs_process_fn:
            obs = self.obs_process_fn(obs)
        model_in = torch.cat([obs, action], dim=obs.ndim - 1)
        return model_in

    def process_batch(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        """The function processes the given batch, normalizes inputs and targets,
         and prepares them for the training.

        Arguments:
            obs (torch.Tensor): the observations tensor
            next_obs (torch.Tensor): the next observation tensor
            action (torch.Tensor): the actions tensor
            reward (torch.Tensor): the rewards tensor

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): the training input and target tensors
        """

        target_obs = next_obs - obs
        for dim in self.no_delta_list:
            target_obs[..., dim] = next_obs[..., dim]

        model_in = self.get_model_input(obs, action)
        if self.learned_rewards:
            target = torch.cat([target_obs, reward], dim=obs.ndim - 1)
        else:
            target = target_obs

        model_in_normalized = self.input_normalizer.normalize(model_in.float(), True)
        target_normalized = self.target_normalizer.normalize(target.float(), True)
        return model_in_normalized, target_normalized

    def save(self, save_dir: str) -> None:
        """Saving the model

        Arguments:
            save_dir (str): the directory to save the model
        """
        self.model.save(save_dir)

    def load(self, load_dir: str) -> None:
        """Loading the model

        Arguments:
            load_dir (str): the directory to load the model
        """
        self.model.load(load_dir)


class DeterministicModel(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: torch.device,
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.device = torch.device(device)

        network = [
            nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
            )
        ]
        for _ in range(num_hidden_layers - 1):
            network.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                )
            )
        network.append(nn.Sequential(nn.Linear(hidden_size, out_size)))
        self.network = nn.Sequential(*network).to(self.device)
        self.network.apply(normal_init)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)

    def loss(
        self,
        model_in: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, any]]:
        prediction = self.forward(model_in)
        loss = F.mse_loss(prediction, target)
        return loss, {"loss_info": None}

    def sample(
        self,
        model_input: torch.Tensor,
        rng: torch.Generator = None,
    ) -> torch.Tensor:
        """Samples next observation, reward and terminal from the model

        Args:
            model_input (tensor): the observation and action.
            rng (torch.Generator): a random number generator.

        Returns:
            (tuple): predicted observation, rewards, terminal indicator and model
                state dictionary.
        """
        return self.forward(model_input)


class Normalizer:
    """Class that keeps a running mean and variance and normalizes data accordingly."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.eps = 1e-5
        self.update_rate = 0.5
        self.bp_step = 0

    def update_stats(self, data: torch.Tensor):
        """Updates the stored statistics using the given data.

        Arguments:
            data (torch.Tensor): The data used to compute the statistics.

        """
        if self.mean is None:
            self.mean = data.mean(0, keepdim=True)
            self.std = data.std(0, keepdim=True)
        else:
            self.mean = (
                1.0 - self.update_rate
            ) * self.mean + self.update_rate * data.mean(0, keepdim=True)
            self.std = (
                1.0 - self.update_rate
            ) * self.std + self.update_rate * data.std(0, keepdim=True)
        self.std[self.std < self.eps] = self.eps
        self.update_rate -= 0.01
        if self.update_rate < 0.01:
            self.update_rate = 0.01

    def normalize(self, val: torch.Tensor, update_state: bool = False) -> torch.Tensor:
        """Normalizes the value according to the stored statistics.

        Arguments:
            val (torch.Tensor): The value to normalize.
            update_state (bool): Update state?

        Returns:
            (torch.Tensor): The normalized value.
        """
        if update_state:
            self.update_stats(val)
        if self.mean is None:
            return val
        return (val - self.mean) / self.std

    def denormalize(self, val: torch.Tensor) -> torch.Tensor:
        """De-normalizes the value according to the stored statistics.

        Arguments:
            val (torch.Tensor): The value to de-normalize.

        Returns:
            (torch.Tensor): The de-normalized value.
        """
        if self.mean is None:
            return val
        return self.std * val + self.mean
