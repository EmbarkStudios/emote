from gettext import npgettext
import os
import time
import isaacgym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from emote.experiments.emit.emit_collector import EmitCollector, EmitCollector2, EmitFeatureAgentProxy, HiveEmitWrapper
from emote.nn.initialization import ortho_init_
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
import torch
from pathlib import Path
import yaml
from torch import nn
from torch.optim import Adam
import gym
from gym.vector import SyncVectorEnv
from emit.train_on_cog import create_cog_hive_config, fixup_configs
from emit.utils.rlgames_utils import get_rlgames_env_creator

import argparse

from emote import Trainer
from emote.callbacks import TensorboardLogger, TerminalLogger
from emote.nn import GaussianPolicyHead
from emote.memory.builder import DictObsTable
from emote.sac import (
    QLoss,
    QTarget,
    PolicyLoss,
    AlphaLoss,
    FeatureAgentProxy,
)
from emote.memory import TableMemoryProxy, MemoryLoader


class QNet(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(obs + act, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, action, obs):
        x = torch.cat([obs, action], dim=1)
        return self.q(x)


class Policy(nn.Module):
    def __init__(self, obs, act):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            GaussianPolicyHead(1024, act),
        ).cuda()

    def forward(self, obs):
        return self.pi(obs)


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="emote")
    parser.add_argument('--env_config', type=str, default="WasabiEnv")
    parser.add_argument('--rl_config', type=str, default="WasabiCogMlp")
    parser.add_argument('--log_dir', type=str, default="/mnt/mllogs")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()
    emit_dir = "/home/jack/Soft/projects/emit"

    config_dir = Path(emit_dir + "/config").resolve()

    file_path = config_dir / (args.env_config + '.yaml')
    task_config = yaml.load(file_path.read_text())

    file_path = config_dir / (args.rl_config + '.yaml')
    train_config = yaml.load(file_path.read_text())
    return args, task_config, train_config


if __name__ == "__main__":
    msg = """
    Embark Modular Isaac Training.

    - Training using emote and Soft Actor Critic
    """
    print("-" * 20)
    print("-" * 20)
    print(msg)
    print("-" * 20)
    print("-" * 20)

    args, task_config, train_config = parse_input_args()

    protocol_config, task_config = fixup_configs(args, train_config, task_config)
    hive_config = create_cog_hive_config(args, task_config, protocol_config)
    hive_config['protocol_config']['task_config'] = task_config

    env = get_rlgames_env_creator(
        task_config=task_config,
        task_name=task_config['name'],
        sim_device=f"cuda:{args.device_id}",
        graphics_device_id=args.device_id,
        headless=task_config['headless'],
    )()

    num_features = train_config['input_shapes']['features']['shape'][0]
    num_actions = train_config['action_count']

    # hive_space = MDPSpace(
    #     BoxSpace(torch.float32, (1,)),
    #     BoxSpace(torch.float32, (num_actions,)),
    #     DictSpace({"obs": BoxSpace(torch.float32, (num_features,))}),
    # )
    hive_space = MDPSpace(
        BoxSpace(np.float32, (1,)),
        BoxSpace(np.float32, (num_actions,)),
        DictSpace({"obs": BoxSpace(np.float32, (num_features,))}),
    )

    table = DictObsTable(spaces=hive_space, maxlen=1_000_000_000)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table, 2000, 5, 500, "batch_size")

    q1 = QNet(num_features, num_actions).to(env.device)
    q2 = QNet(num_features, num_actions).to(env.device)
    policy = Policy(num_features, num_actions).to(env.device)

    # q1.apply(ortho_init_)
    # q2.apply(ortho_init_)
    # policy.apply(ortho_init_)

    ln_alpha = torch.tensor(1.0, requires_grad=True)  # .to(env.device)
    agent_proxy = EmitFeatureAgentProxy(policy)

    learning_rate = 1e-4
    max_grad_norm = 0.5

    logged_cbs = [
        QLoss(name="q1", q=q1, opt=Adam(q1.parameters(), lr=learning_rate), max_grad_norm=max_grad_norm),
        QLoss(name="q2", q=q2, opt=Adam(q2.parameters(), lr=learning_rate), max_grad_norm=max_grad_norm),
        PolicyLoss(pi=policy, ln_alpha=ln_alpha, q=q1, opt=Adam(policy.parameters(), lr=learning_rate), max_grad_norm=max_grad_norm),
        AlphaLoss(pi=policy, ln_alpha=ln_alpha, opt=Adam([ln_alpha], lr=learning_rate), n_actions=1, max_grad_norm=max_grad_norm),
        QTarget(pi=policy, ln_alpha=ln_alpha, q1=q1, q2=q2, gamma=0.95),
    ]

    callbacks = logged_cbs + [
        EmitCollector2(env, agent_proxy, memory_proxy),
        TerminalLogger(logged_cbs, 10),
        TensorboardLogger(
            logged_cbs,
            SummaryWriter(log_dir=args.log_dir + "/" + args.experiment + '_{}'.format(time.time())),
            step_log_interval=10,
            samples_log_interval=10_000_000,
        )
    ]

    trainer = Trainer(callbacks, dataloader, 200)
    trainer.train()
