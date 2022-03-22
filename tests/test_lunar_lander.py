import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.optim import Adam
from gym.vector import AsyncVectorEnv, SyncVectorEnv
import gym
import numpy as np

from emote import Trainer
from emote.callbacks import (
    FinalLossTestCheck,
    TensorboardLogger
)
from emote.nn import GaussianMLPPolicy, ActionValue
from emote.memory.builder import DictObsTable
from emote.sac import (
    QLoss,
    QTarget,
    PolicyLoss,
    AlphaLoss,
    FeatureAgentProxy,
)
from emote.memory import TableMemoryProxy, MemoryLoader

from .gym import SimpleGymCollector, DictGymWrapper


N_HIDDEN = 256


def test_lunar_lander():

    experiment_name = "Lunar-lander_test2"

    hidden_layers = [256, 256]

    batch_size = 500
    rollout_len = 2

    n_env = 60

    learning_rate = 1e-3

    env = DictGymWrapper(SyncVectorEnv([_make_env(i) for i in range(n_env)]))
    table = DictObsTable(spaces=env.dict_space, maxlen=4_000_000)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table, batch_size, rollout_len, "batch_size")

    num_actions = env.dict_space.actions.shape[0]
    num_obs = list(env.dict_space.state.spaces.values())[0].shape[0]

    q1 = ActionValue(num_obs, num_actions, hidden_layers)
    q2 = ActionValue(num_obs, num_actions, hidden_layers)
    policy = GaussianMLPPolicy(num_obs, num_actions, hidden_layers)

    ln_alpha = torch.tensor(1.0, requires_grad=True)
    agent_proxy = FeatureAgentProxy(policy)

    logged_cbs = [
        QLoss(name="q1", q=q1, opt=Adam(q1.parameters(), lr=learning_rate)),
        QLoss(name="q2", q=q2, opt=Adam(q2.parameters(), lr=learning_rate)),
        PolicyLoss(pi=policy, ln_alpha=ln_alpha, q=q1, opt=Adam(policy.parameters(), lr=learning_rate)),
        AlphaLoss(pi=policy, ln_alpha=ln_alpha, opt=Adam([ln_alpha]), n_actions=num_actions),
        QTarget(pi=policy, ln_alpha=ln_alpha, q1=q1, q2=q2),
    ]

    callbacks = logged_cbs + [
        SimpleGymCollector(env, agent_proxy, memory_proxy, warmup_steps=batch_size*rollout_len),
        TensorboardLogger(logged_cbs, SummaryWriter("runs/"+experiment_name), 2000),
        FinalLossTestCheck([logged_cbs[2]], [10.0], 300_000_000),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()

def _make_env(rank):
    def _thunk():
        env = gym.make("LunarLander-v2", continuous=True)
        env.seed(rank)
        return env
    return _thunk
    
