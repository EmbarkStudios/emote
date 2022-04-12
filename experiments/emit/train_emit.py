import argparse
from functools import partial
import time
from pathlib import Path

import isaacgym
import numpy as np
import torch
import yaml
from emote import Trainer
from emote.callbacks import TensorboardLogger
from emote.collectors.emit_collector import (
    EmitAgentProxy,
    EmitCollector,
    EmitWrapper,
)
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsNStepTable
from emote.nn import GaussianPolicyHead
from emote.nn.initialization import ortho_init_, xavier_uniform
from emote.sac import AlphaLoss, PolicyLoss, QLoss, QTarget
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from emit.utils.env import env_creator


def parse_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="wasabi")
    parser.add_argument("--emit_dir", type=str, required=True)
    parser.add_argument("--env_config", type=str, default="WasabiEnv")
    parser.add_argument("--rl_config", type=str, default="WasabiEmoteMlp")
    parser.add_argument("--log_dir", type=str, default="/mnt/mllogs/emote/emit")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

    config_dir = Path(args.emit_dir + "/config").resolve()

    file_path = config_dir / (args.env_config + ".yaml")
    task_config = yaml.load(file_path.read_text(), Loader=yaml.Loader)

    file_path = config_dir / (args.rl_config + ".yaml")
    train_config = yaml.load(file_path.read_text(), Loader=yaml.Loader)
    return args, task_config, train_config


def fixup_configs(args, train_config, task_config):
    use_gpu = not args.cpu
    if use_gpu:
        full_device_str = f"cuda:{args.device_id}"
    else:
        full_device_str = "cpu"
    task_config["sim"]["use_gpu_pipeline"] = use_gpu
    task_config["sim"]["physx"]["use_gpu"] = use_gpu
    task_config["device_id"] = args.device_id
    task_config["rl_device"] = full_device_str
    task_config["sim_device"] = full_device_str
    task_config["headless"] = not (args.infer or args.render)
    if args.infer:
        task_config["env"]["num_envs"] = train_config["isaac"]["num_inf_envs"]
        task_config["env"]["env_spacing"] = 200
    else:
        task_config["env"]["num_envs"] = train_config["isaac"]["num_envs"]
        task_config["env"]["viewer"]["ref_env_start_stop_step"] = [0, 1, 1]
        task_config["env"]["env_spacing"] = 60

    if train_config["isaac"]["protocol_kind"] == "sacmlp":
        task_config["env"]["vision"]["flatten"] = True
    else:
        assert False
    return task_config


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
        self.policy = GaussianPolicyHead(hidden_dims[-1], num_actions)

        self.encoder.apply(ortho_init_)
        # self.policy.apply(partial(ortho_init_, gain=1))
        self.policy.apply(partial(xavier_uniform, gain=0.01))

    def forward(self, obs):
        return self.policy(self.encoder(obs))


if __name__ == "__main__":
    msg = """
    Embark Modular Isaac Training.

    - Training using emote and Soft Actor Critic
    """
    print("-" * 20)
    print(msg)
    print("-" * 20)
    device = torch.device("cuda")

    args, task_config, train_config = parse_input_args()
    task_config = fixup_configs(args, train_config, task_config)

    num_features = (
        task_config["env"]["num_observations"]
        + task_config["env"]["vision"]["width"] ** 2
    )
    num_actions = task_config["env"]["num_actions"]
    hidden_dims = train_config["hidden_sizes"]
    rollout_len = train_config["rollout_length"]
    batch_size = train_config["batch_size"]
    learning_rate = train_config["learning_rate"]
    max_grad_norm = train_config["max_grad_norm"]
    gamma = train_config["gamma"]
    num_envs = task_config["env"]["num_envs"]
    init_alpha = train_config["init_alpha"]

    emit_space = MDPSpace(
        BoxSpace(np.float32, (1,)),
        BoxSpace(np.float32, (num_actions,)),
        DictSpace({"obs": BoxSpace(np.float32, (num_features,))}),
    )

    table = DictObsNStepTable(
        spaces=emit_space,
        use_terminal_column=True,
        maxlen=train_config["memory_max_size"],
        device=device,
    )
    memory_proxy = TableMemoryProxy(table, use_terminal=True)
    dataloader = MemoryLoader(
        table, batch_size // rollout_len, rollout_len, "batch_size"
    )

    q1 = QNet(num_features, num_actions, hidden_dims).to(device)
    q2 = QNet(num_features, num_actions, hidden_dims).to(device)
    policy = Policy(num_features, num_actions, hidden_dims).to(device)

    ln_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=device)
    agent_proxy = EmitAgentProxy(policy, device=device)

    q1 = q1.to(device)
    q2 = q2.to(device)
    policy = policy.to(device)

    env = env_creator(
        task_config=task_config,
        task_name=task_config["name"],
        sim_device=f"cuda:{args.device_id}",
        graphics_device_id=args.device_id,
        headless=task_config["headless"],
    )()

    env = EmitWrapper(env, device, has_images=False)

    eps = 1e-8

    logged_cbs = [
        QLoss(
            name="q1",
            q=q1,
            opt=Adam(q1.parameters(), lr=learning_rate, eps=eps),
            max_grad_norm=max_grad_norm,
        ),
        QLoss(
            name="q2",
            q=q2,
            opt=Adam(q2.parameters(), lr=learning_rate, eps=eps),
            max_grad_norm=max_grad_norm,
        ),
        PolicyLoss(
            pi=policy,
            ln_alpha=ln_alpha,
            q=q1,
            opt=Adam(policy.parameters(), lr=learning_rate, eps=eps),
            max_grad_norm=max_grad_norm,
        ),
        AlphaLoss(
            pi=policy,
            ln_alpha=ln_alpha,
            opt=Adam([ln_alpha], lr=learning_rate, eps=eps),
            n_actions=num_actions,
            max_grad_norm=max_grad_norm,
            max_alpha=0.1,
        ),
        QTarget(
            pi=policy,
            ln_alpha=ln_alpha,
            q1=q1,
            q2=q2,
            gamma=gamma,
            roll_length=rollout_len,
        ),
        EmitCollector(
            env,
            agent_proxy,
            memory_proxy,
            warmup_steps=batch_size * 2,
            inf_steps_per_bp=batch_size // num_envs,  # Aim for 1:1 data reuse
        ),
    ]

    callbacks = logged_cbs + [
        TensorboardLogger(
            logged_cbs,
            SummaryWriter(
                log_dir=args.log_dir + "/" + args.experiment + "_{}".format(time.time())
            ),
            log_interval=100,
        ),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()
