from collections import deque
from typing import Optional

import torch

from emote.callback import Callback
from emote.callbacks import LoggingMixin
from emote.memory import MemoryLoader
from emote.models.model_env import ModelEnv
from emote.proxies import AgentProxy, MemoryProxy
from emote.typing import AgentId, BPStepScheduler, DictObservation
from emote.utils.math import truncated_linear


class BatchCallback(LoggingMixin, Callback):
    def __init__(self):
        super().__init__()

    def begin_batch(self, *args, **kwargs):
        pass

    @Callback.extend
    def collect_multiple(self, *args, **kwargs):
        pass


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
        self.prob_of_sampling_model_data = truncated_linear(
            min_x=self.scheduler.bp_step_begin,
            max_x=self.scheduler.bp_step_end,
            min_y=self.scheduler.value_min,
            max_y=self.scheduler.value_max,
            x=self.bp_counter,
        )
        rnd = torch.rand(size=(1,), generator=self.rng)[0]
        return True if rnd < self.prob_of_sampling_model_data else False


class ModelBasedCollector(BatchCallback):
    def __init__(
        self,
        model_env: ModelEnv,
        agent: AgentProxy,
        memory: MemoryProxy,
        rollout_scheduler: BPStepScheduler,
        num_bp_to_retain_buffer=1000000,
        data_group: str = "default",
    ):
        super().__init__()
        """The data group is used to receive correct observation when collect_multiple is 
        called. The data group must be chosen such that real Gym samples (not model data) 
        are given to the function. """
        self.data_group = data_group

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
        self.collect_multiple(*args, **kwargs)

    def collect_multiple(self, observation):
        """Collect multiple rollouts
        :param observation: initial observations
        """
        self.obs = self.model_env.dict_reset(observation["obs"], self.len_rollout)
        for _ in range(self.len_rollout + 1):
            self.collect_sample()

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
        len_rollout = int(
            truncated_linear(
                min_x=self.rollout_scheduler.bp_step_begin,
                max_x=self.rollout_scheduler.bp_step_end,
                min_y=self.rollout_scheduler.value_min,
                max_y=self.rollout_scheduler.value_max,
                x=self.bp_counter,
            )
        )
        if self.len_rollout != len_rollout:
            self.len_rollout = len_rollout
            new_memory_size = (
                self.len_rollout
                * self.model_env.num_envs
                * self.num_bp_to_retain_buffer
            )
            self.memory.resize(new_memory_size)
