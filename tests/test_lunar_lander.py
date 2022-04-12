import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.optim import Adam
from gym.vector import AsyncVectorEnv
import gym
import time

from emote import Trainer
from emote.callbacks import FinalLossTestCheck, TensorboardLogger
from emote.nn import GaussianPolicyHead
from emote.nn.initialization import ortho_init_
from emote.memory.builder import DictObsNStepTable
from emote.sac import (
    QLoss,
    QTarget,
    PolicyLoss,
    AlphaLoss,
    FeatureAgentProxy,
)
from emote.memory import TableMemoryProxy, MemoryLoader
from tests.gym.collector import ThreadedGymCollector
from tests.gym import DictGymWrapper


class QNet(nn.Module):
    def __init__(self, num_obs, num_actions, num_hidden):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(num_obs + num_actions, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1),
        )
        self.q.apply(ortho_init_)

    def forward(self, action, obs):
        x = torch.cat([obs, action], dim=1)
        return self.q(x)


class Policy(nn.Module):
    def __init__(self, num_obs, num_actions, num_hidden):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(num_obs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            GaussianPolicyHead(num_hidden, num_actions),
        )
        self.pi.apply(ortho_init_)

    def forward(self, obs):
        return self.pi(obs)


def test_lunar_lander():
    device = torch.device("cuda")

    experiment_name = "lunar_lander_test_" + str(time.time())

    hidden_layer = 256
    batch_size = 2000
    rollout_len = 20
    n_env = 10
    learning_rate = 5e-3
    max_grad_norm = 1

    env = DictGymWrapper(AsyncVectorEnv([_make_env(i) for i in range(n_env)]))
    table = DictObsNStepTable(
        spaces=env.dict_space,
        use_terminal_column=False,
        maxlen=4_000_000,
        device=device,
    )
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(
        table, batch_size // rollout_len, rollout_len, "batch_size"
    )

    num_actions = env.dict_space.actions.shape[0]
    num_obs = list(env.dict_space.state.spaces.values())[0].shape[0]

    q1 = QNet(num_obs, num_actions, hidden_layer)
    q2 = QNet(num_obs, num_actions, hidden_layer)
    policy = Policy(num_obs, num_actions, hidden_layer)

    ln_alpha = torch.tensor(1.0, requires_grad=True, device=device)
    agent_proxy = FeatureAgentProxy(policy, device=device)

    q1 = q1.to(device)
    q2 = q2.to(device)
    policy = policy.to(device)

    logged_cbs = [
        QLoss(
            name="q1",
            q=q1,
            opt=Adam(q1.parameters(), lr=learning_rate),
            max_grad_norm=max_grad_norm,
        ),
        QLoss(
            name="q2",
            q=q2,
            opt=Adam(q2.parameters(), lr=learning_rate),
            max_grad_norm=max_grad_norm,
        ),
        PolicyLoss(
            pi=policy,
            ln_alpha=ln_alpha,
            q=q1,
            opt=Adam(policy.parameters(), lr=learning_rate),
            max_grad_norm=max_grad_norm,
        ),
        AlphaLoss(
            pi=policy,
            ln_alpha=ln_alpha,
            opt=Adam([ln_alpha]),
            n_actions=num_actions,
            max_grad_norm=max_grad_norm,
        ),
        QTarget(
            pi=policy,
            ln_alpha=ln_alpha,
            q1=q1,
            q2=q2,
            roll_length=rollout_len,
        ),
        ThreadedGymCollector(
            env,
            agent_proxy,
            memory_proxy,
            warmup_steps=batch_size * rollout_len,
            render=False,
        ),
    ]

    callbacks = logged_cbs + [
        TensorboardLogger(logged_cbs, SummaryWriter("runs/" + experiment_name), 100),
        FinalLossTestCheck([logged_cbs[2]], [10.0], 50_000_000),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()


def _make_env(rank):
    def _thunk():
        env = gym.make("LunarLander-v2", continuous=True)
        env = gym.wrappers.FrameStack(env, 3)
        env = gym.wrappers.FlattenObservation(env)
        env.seed(rank)
        return env

    return _thunk
