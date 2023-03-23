import argparse
import time

from functools import partial

import gymnasium as gym
import numpy as np
import torch
import wandb

from gymnasium.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper
from tests.gym.collector import ThreadedGymCollector
from torch import nn
from torch.optim import Adam

from emote import Trainer
from emote.callbacks import BackPropStepsTerminator, WBLogger
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsNStepTable
from emote.nn import GaussianPolicyHead
from emote.nn.initialization import ortho_init_, xavier_uniform_init_
from emote.sac import AlphaLoss, FeatureAgentProxy, PolicyLoss, QLoss, QTarget


def _make_env():
    def _thunk():
        env = gym.make("LunarLander-v2", continuous=True)
        env = gym.wrappers.FrameStack(env, 3)
        env = gym.wrappers.FlattenObservation(env)
        return env

    return _thunk


class QNet(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dims):
        super().__init__()
        all_dims = [num_obs + num_actions] + hidden_dims

        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip(all_dims, hidden_dims)
            ],
        )
        self.encoder.apply(ortho_init_)

        self.final_layer = nn.Linear(hidden_dims[-1], 1)
        self.final_layer.apply(partial(ortho_init_, gain=1))

    def forward(self, action, obs):
        x = torch.cat([obs, action], dim=1)
        return self.final_layer(self.encoder(x))


class Policy(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dims):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip([num_obs] + hidden_dims, hidden_dims)
            ],
        )
        self.policy = GaussianPolicyHead(
            hidden_dims[-1],
            num_actions,
        )

        self.encoder.apply(ortho_init_)
        self.policy.apply(partial(xavier_uniform_init_, gain=0.01))

    def forward(self, obs):
        sample, log_prob = self.policy(self.encoder(obs))
        # TODO: Investigate the log_prob() logic of the pytorch distribution code.
        # The change below shouldn't be needed but significantly improves training
        # stability when training lunar lander.
        log_prob = log_prob.clamp(min=-2)
        return sample, log_prob


def train_lunar_lander(args):
    device = torch.device(args.device)

    hidden_dims = [256, 256]
    batch_size = 2000
    n_env = 10
    max_grad_norm = 1

    rollout_len = 20
    init_alpha = 1.0

    # any additional hyperparameters/metadata that we want to log
    config = {
        "hidden_dims": hidden_dims,
        "batch_size": batch_size,
        "rollout_len": rollout_len,
    }

    # parameters to search defined from wandb.config which are set by the sweep agent
    wandb.init(config=config)
    learning_rate = wandb.config.learning_rate

    env = DictGymWrapper(AsyncVectorEnv([_make_env() for _ in range(n_env)]))
    table = DictObsNStepTable(
        spaces=env.dict_space,
        use_terminal_column=False,
        maxlen=4_000_000,
        device=device,
    )
    memory_proxy = TableMemoryProxy(table, use_terminal=False)
    dataloader = MemoryLoader(
        table, batch_size // rollout_len, rollout_len, "batch_size"
    )

    num_actions = env.dict_space.actions.shape[0]
    num_obs = list(env.dict_space.state.spaces.values())[0].shape[0]

    q1 = QNet(num_obs, num_actions, hidden_dims)
    q2 = QNet(num_obs, num_actions, hidden_dims)
    policy = Policy(num_obs, num_actions, hidden_dims)

    ln_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=device)
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
            opt=Adam([ln_alpha], lr=learning_rate),
            n_actions=num_actions,
            max_grad_norm=max_grad_norm,
            max_alpha=10.0,
        ),
        QTarget(
            pi=policy,
            ln_alpha=ln_alpha,
            q=[q1, q2],
            roll_length=rollout_len,
            reward_scale=0.1,
        ),
        ThreadedGymCollector(
            env,
            agent_proxy,
            memory_proxy,
            warmup_steps=batch_size,
            render=False,
        ),
    ]

    logger = WBLogger(
        callbacks=logged_cbs,
        config=config,
        log_interval=100,
    )

    callbacks = logged_cbs + [logger, BackPropStepsTerminator(args.num_bp_steps)]
    trainer = Trainer(callbacks, dataloader)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ll")
    parser.add_argument("--log-dir", type=str, default="/mnt/mllogs/emote/lunar_lander")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_bp_steps", type=int, default=10000)
    parser.add_argument(
        "--wandb_run",
        type=str,
        default=None,
        help="Short display name of run for the W&B UI. Randomly generated by default.",
    )

    args = parser.parse_args()

    # Configuration dictionary of the W&B sweep
    sweep_config = {
        "method": "grid",
        "name": "sweep",
        "parameters": {
            "learning_rate": {"values": [8e-3, 1e-3]},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project=args.name)
    wandb.agent(sweep_id, function=partial(train_lunar_lander, args))
