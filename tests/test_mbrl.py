import torch

from gymnasium.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper, HitTheMiddle, SimpleGymCollector
from torch import nn
from torch.optim import Adam

from emote import Trainer
from emote.callbacks import BackPropStepsTerminator
from emote.extra.schedules import BPStepScheduler
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsTable
from emote.models.callbacks import LossProgressCheck, ModelBasedCollector, ModelLoss
from emote.models.ensemble import EnsembleOfGaussian
from emote.models.model import DynamicModel
from emote.models.model_env import ModelEnv
from emote.sac import FeatureAgentProxy
from emote.utils.spaces import MDPSpace


class FakeDataloader:
    def __init__(self, data_group: str, num_obs: int, batch_size: int):
        self.data_group = data_group
        self.num_obs = num_obs
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            batch = {
                "observation": {"obs": torch.rand(self.batch_size, self.num_obs)},
            }
            yield {self.data_group: batch, "batch_size": self.batch_size}


class MultiplierNN(nn.Module):
    def __init__(self, value: float, device: torch.device):
        super().__init__()
        self.value = value
        self.device = device

    def forward(self, x: torch.Tensor):
        return self.value * x

    def sample(self, x, rng):
        return self.forward(x)


class RandomPolicy(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        rand_actions = 2 * (torch.rand(batch_size, self.action_dim) - 0.5)
        return rand_actions, 0


def create_memory(
    space: MDPSpace,
    memory_size: int,
    len_rollout: int,
    batch_size: int,
    data_group: str,
    device: torch.device,
):
    """Creates memory and data_loader for the RL training

    Arguments:
        space (MDPSpace): the MDP space
        memory_size (int): the maximum length of memory
        len_rollout (int): the rollout size for the NStepTable
        batch_size (int): batch size
        data_group (str): the data group for uploading the data
        device (torch.device): the device to upload the data
    Returns:
        (tuple[TableMemoryProxy, MemoryLoader]): A proxy for the memory and a dataloader

    """
    table = DictObsTable(
        spaces=space,
        use_terminal_column=False,
        maxlen=memory_size,
        device=device,
    )
    memory_proxy = TableMemoryProxy(table=table, use_terminal=False)
    data_loader = MemoryLoader(
        table=table,
        rollout_count=batch_size // len_rollout,
        rollout_length=len_rollout,
        size_key="batch_size",
        data_group=data_group,
    )
    return memory_proxy, data_loader


NUM_OBS = 2
NUM_ACTIONS = 1
RL_DATA_GROUP = "rl_group"


def test_model_collector():
    """The function tests unrolling a dynamic model and storing the rollouts in a replay buffer.
    The fake dynamic model simply multiplies the inputs by a fixed (rand) number, i.e.,
        next_obs = obs x rand_number,
        rewards = actions x rand_number.

    The test checks the following:
        * the replay buffer contains a correct number of samples,
        * stored samples are the ones generated by the fake model.

    """
    batch_size = 10
    rollout_size = 5
    rand_multiplier = torch.rand(1)[0] * 10
    env = DictGymWrapper(AsyncVectorEnv(2 * [HitTheMiddle]))  # dummy envs
    device = torch.device("cpu")
    model = MultiplierNN(value=rand_multiplier, device=device)
    dynamic_model = DynamicModel(model=model, no_delta_list=[0, 1])
    model_env = ModelEnv(
        num_envs=batch_size,
        model=dynamic_model,
        termination_fn=lambda a: torch.zeros(a.shape[0]),
    )
    policy = RandomPolicy(action_dim=NUM_ACTIONS)
    agent = FeatureAgentProxy(policy, device)
    memory, dataloader = create_memory(
        env.dict_space,
        memory_size=100,
        len_rollout=1,
        batch_size=batch_size,
        data_group=RL_DATA_GROUP,
        device=device,
    )
    callbacks = [
        ModelBasedCollector(
            model_env=model_env,
            agent=agent,
            memory=memory,
            rollout_scheduler=BPStepScheduler(*[0, 10, rollout_size, rollout_size]),
        ),
        BackPropStepsTerminator(bp_steps=1),
    ]
    fake_dataset = FakeDataloader(num_obs=NUM_OBS, data_group="default", batch_size=batch_size)
    trainer = Trainer(callbacks, fake_dataset)
    trainer.train()

    if memory.size() != (rollout_size * batch_size):
        raise Exception(
            f"The RL replay buffer must contain rollout_size x batch_size "
            f"= {rollout_size * batch_size} but it contains {memory.size()}"
        )

    data_iter = iter(dataloader)
    batch = next(data_iter)

    if RL_DATA_GROUP not in batch.keys():
        raise Exception("The RL data group does not exist in the keys\n")
    batch = batch[RL_DATA_GROUP]

    model_in = torch.cat((batch["observation"]["obs"], batch["actions"]), dim=1)
    model_out = torch.cat((batch["next_observation"]["obs"], batch["rewards"]), dim=1)

    if torch.mean(torch.abs(rand_multiplier * model_in - model_out)) > 0.001:
        raise Exception("The loaded samples do not look correct.")


def test_ensemble_training():
    """The function tests ensemble training. The test will pass if the loss goes down according
    to the given criterion.
    """
    device = torch.device("cpu")
    batch_size = 200
    rollout_length = 1
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(spaces=env.dict_space, maxlen=10000, device=device)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(
        table=table,
        rollout_count=batch_size // rollout_length,
        rollout_length=rollout_length,
        size_key="batch_size",
    )

    model = EnsembleOfGaussian(
        in_size=NUM_OBS + NUM_ACTIONS,
        out_size=NUM_OBS + 1,
        device=device,
        ensemble_size=5,
    )
    dynamic_model = DynamicModel(model=model)
    policy = RandomPolicy(action_dim=1)
    agent_proxy = FeatureAgentProxy(policy, device)

    callbacks = [
        ModelLoss(model=dynamic_model, opt=Adam(dynamic_model.model.parameters())),
        LossProgressCheck(model=dynamic_model, num_bp=500),
        SimpleGymCollector(env, agent_proxy, memory_proxy, warmup_steps=500, render=False),
    ]
    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    env.close()
