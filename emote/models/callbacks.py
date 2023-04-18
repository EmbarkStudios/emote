from collections import deque
from typing import Optional

import numpy as np
import torch

from torch import optim

from emote.callbacks import BatchCallback, LossCallback
from emote.extra.schedules import BPStepScheduler
from emote.memory import MemoryLoader
from emote.models.model import DynamicModel
from emote.models.model_env import ModelEnv
from emote.proxies import AgentProxy, MemoryProxy
from emote.trainer import TrainingShutdownException
from emote.typing import AgentId, DictObservation


class ModelLoss(LossCallback):
    """Trains a dynamic model by minimizing the model loss

    Arguments:
        dynamic_model (DynamicModel): A dynamic model
        opt (torch.optim.Optimizer): An optimizer.
        lr_schedule (lr_scheduler, optional): A learning rate scheduler
        max_grad_norm (float): Clip the norm of the gradient during backprop using this value.
        name (str): The name of the module. Used e.g. while logging.
        data_group (str): The name of the data group from which this Loss takes its data.
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
        input_key: str = "obs",
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
        self._input_key = input_key

    def loss(self, observation, next_observation, actions, rewards):
        loss, _ = self.model.loss(
            obs=observation[self._input_key],
            next_obs=next_observation[self._input_key],
            action=actions,
            reward=rewards,
        )
        return loss


class LossProgressCheck(BatchCallback):
    def __init__(
        self,
        model: DynamicModel,
        num_bp: int,
        data_group: str = "default",
        input_key: str = "obs",
    ):
        super().__init__()
        self.data_group = data_group
        self.model = model
        self.cycle = num_bp
        self.rng = torch.Generator(device=self.model.device)
        self.prediction_err = []
        self.prediction_average_err = []
        self.len_averaging_window = num_bp // 10
        self._input_key = input_key

    def begin_batch(self, *args, **kwargs):
        obs, next_obs, action, reward = self.get_batch(*args, **kwargs)
        predicted_obs, predicted_reward = self.model.sample(
            observation=obs, action=action, rng=self.rng
        )
        obs_prediction_err = (predicted_obs - next_obs).detach().to("cpu").numpy()
        reward_prediction_err = (predicted_reward - reward).detach().to("cpu").numpy()

        obs_prediction_err = np.mean(np.abs(obs_prediction_err))
        reward_prediction_err = np.mean(np.abs(reward_prediction_err))

        self.log_scalar("mbrl/obs_pred_err", obs_prediction_err)
        self.log_scalar("mbrl/rew_pred_err", reward_prediction_err)

        self.prediction_err.append([obs_prediction_err, reward_prediction_err])

        if len(self.prediction_err) >= self.len_averaging_window:
            self.prediction_average_err.append(
                np.mean(np.array(self.prediction_err), axis=0)
            )
            self.prediction_err = []

    def end_cycle(self):
        for i in range(len(self.prediction_average_err) - 4):
            for j in range(2):
                if (
                    self.prediction_average_err[i + 4][j]
                    > self.prediction_average_err[i][j]
                ):
                    raise Exception(
                        f"The loss is not decreasing: \n"
                        f"Loss at {i}: {self.prediction_average_err[i]}"
                        f"Loss at {i+4}: {self.prediction_average_err[i+4]}"
                    )
        raise TrainingShutdownException()

    def get_batch(self, observation, next_observation, actions, rewards):
        return (
            observation[self._input_key],
            next_observation[self._input_key],
            actions,
            rewards,
        )


class BatchSampler(BatchCallback):
    """BatchSampler class is used to provide batches of data for the RL training callbacks.
    In every BP step, it samples one batch from either the gym buffer or the model buffer
    based on a Bernoulli probability distribution. It outputs the batch to a separate
    data-group which will be used by other RL training callbacks.

        Arguments:
            dataloader (MemoryLoader): the dataloader to load data from the model buffer
            prob_scheduler (BPStepScheduler): the scheduler to update the prob of data
            samples to come from the model vs. the Gym buffer
            data_group (str): the data_group to receive data
            rl_data_group (str): the data_group to upload data for RL training
            generator (torch.Generator (optional)): an optional random generator
    """

    def __init__(
        self,
        dataloader: MemoryLoader,
        prob_scheduler: BPStepScheduler,
        data_group: str = "default",
        rl_data_group: str = "rl_buffer",
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.dataloader = dataloader
        """There are two data_groups in this class: self.data_group, and self.rl_data_group.
        The first one is to receive samples from the Gym buffer in clone_batch method. The
        second one is the destination for the batch. """
        self.data_group = data_group
        self.rl_data_group = rl_data_group
        self.iter = iter(self.dataloader)
        self.scheduler = prob_scheduler
        self.prob_of_sampling_model_data = self.scheduler.value_min
        self.rng = generator if generator else torch.Generator()
        self.bp_counter = 0

    def begin_batch(self, *args, **kwargs):
        """Generates a batch of data either by sampling from the model buffer or by
        cloning the input batch
        Returns:
            (dict): the batch of data
        """
        self.log_scalar(
            "training/prob_sampling_from_model", self.prob_of_sampling_model_data
        )
        if self.use_model_batch():
            return self.sample_model_batch()
        else:
            return {self.rl_data_group: kwargs[self.data_group]}

    def sample_model_batch(self):
        """Samples a batch of data from the model buffer
        Returns:
            (dict): batch samples
        """
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            batch = next(self.iter)
        return batch

    def use_model_batch(self):
        """Decides if batch should come from the model-generated buffer
        Returns:
            (bool): True if model samples should be used, False otherwise.
        """
        self.bp_counter += 1
        self.prob_of_sampling_model_data = self.scheduler.evaluate_at(self.bp_counter)
        rnd = torch.rand(size=(1,), generator=self.rng)[0]
        return True if rnd < self.prob_of_sampling_model_data else False


class ModelBasedCollector(BatchCallback):
    """ModelBasedCollector class is used to sample rollouts from the trained dynamic model.
    The rollouts are stored in a replay buffer memory.

        Arguments:
            model_env: The Gym-like dynamic model
            agent: The policy used to sample actions
            memory: The memory to store the new synthetic samples
            rollout_scheduler: A scheduler used to set the rollout-length when unrolling the dynamic model
            num_bp_to_retain_buffer: The number of BP steps to keep samples. Samples will be over-written (first in
            first out) for bp steps larger than this.
            data_group: The data group to receive data from. This must be set to get real (Gym) samples

    """

    def __init__(
        self,
        model_env: ModelEnv,
        agent: AgentProxy,
        memory: MemoryProxy,
        rollout_scheduler: BPStepScheduler,
        num_bp_to_retain_buffer=1000000,
        data_group: str = "default",
        input_key: str = "obs",
    ):
        super().__init__()
        """ The data group is used to receive correct observation when collect_multiple is 
            called. The data group must be set such that real Gym samples (not model data) 
            are given to the function. 
        """
        self.data_group = data_group
        self._input_key = input_key
        self.agent = agent
        self.memory = memory
        self.model_env = model_env
        self.last_environment_rewards = deque(maxlen=1000)

        self.len_rollout = int(rollout_scheduler.value_min)
        self.rollout_scheduler = rollout_scheduler
        self.num_bp_to_retain_buffer = num_bp_to_retain_buffer
        self.obs: dict[AgentId, DictObservation] = None
        self.prob_of_sampling_model_data = 0.0
        self.bp_counter = 0

    def begin_batch(self, *args, **kwargs):
        self.update_rollout_size()
        self.log_scalar("training/model_rollout_length", self.len_rollout)
        observation = self.get_batch(*args, **kwargs)

        self.obs = self.model_env.dict_reset(observation, self.len_rollout)
        for _ in range(self.len_rollout + 1):
            self.collect_sample()

    def get_batch(self, observation):
        return observation[self._input_key]

    def collect_sample(self):
        """Collect a single rollout"""
        actions = self.agent(self.obs)
        next_obs, ep_info = self.model_env.dict_step(actions)

        self.memory.add(self.obs, actions)
        self.obs = next_obs

        if "reward" in ep_info:
            self.log_scalar("episode/model_reward", ep_info["reward"])

    def update_rollout_size(self):
        self.bp_counter += 1
        len_rollout = int(self.rollout_scheduler.evaluate_at(self.bp_counter))
        if self.len_rollout != len_rollout:
            self.len_rollout = len_rollout
            new_memory_size = (
                self.len_rollout
                * self.model_env.num_envs
                * self.num_bp_to_retain_buffer
            )
            self.memory.resize(new_memory_size)
