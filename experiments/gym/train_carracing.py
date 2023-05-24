import argparse
import time

from dataclasses import dataclass
from functools import partial
from typing import Tuple

import numpy as np
import torch

from gym.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper
from tests.gym.collector import ThreadedGymCollector
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from emote import Trainer
from emote.callback import Callback
from emote.callbacks.logging import TensorboardLogger
from emote.env.box2d import make_vision_box2d_env
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsNStepTable
from emote.mixins.logging import LoggingMixin
from emote.nn import GaussianPolicyHead
from emote.nn.action_value_mlp import SharedEncoderActionValueNet
from emote.nn.initialization import ortho_init_, xavier_uniform_init_
from emote.nn.layers import Conv2dEncoder
from emote.sac import AlphaLoss, PolicyLoss, QLoss, QTarget, VisionAgentProxy


class Policy(nn.Module):
    def __init__(self, shared_enc, num_obs, num_actions, hidden_dims):
        super().__init__()
        self.shared_enc = shared_enc
        self.mlp_encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip([num_obs] + hidden_dims, hidden_dims)
            ],
        )
        self.policy = GaussianPolicyHead(
            hidden_dims[-1],
            num_actions,
        )

        self.mlp_encoder.apply(ortho_init_)
        self.policy.apply(partial(xavier_uniform_init_, gain=0.01))

    def forward(self, obs):
        x = self.shared_enc(obs)
        x = self.mlp_encoder(x)
        sample, log_prob = self.policy(x)
        log_prob = log_prob.clamp(min=-2)
        return sample, log_prob

    def non_shared_parameters(self):
        # ** This is critical for training! **
        # Prevent the policy loss from training the shared encoder.
        return list(self.mlp_encoder.parameters()) + list(self.policy.parameters())


class ImageLoggerCallback(LoggingMixin, Callback):
    def __init__(self):
        super().__init__()
        self.data_group = "default"

    def begin_batch(self, observation):
        self.log_image("images/obs", observation["obs"][0])


@dataclass
class Config:
    device: str = "cuda"
    env_name: str = "CarRacing-v1"
    hidden_dims = [512, 512]
    batch_size: int = 1000
    rollout_len: int = 40
    learning_rate: float = 2e-3
    n_env: int = 10
    max_grad_norm: float = 1.0
    init_alpha: float = 1.0
    max_memory_size: int = 100_000
    max_alpha: float = 10.0
    # Conv encoder
    input_shape: Tuple[int, int, int] = (84, 84, 3)
    channels = [16, 16, 32, 32]
    kernels = [3, 3, 3, 3]
    strides = [2, 2, 2, 2]
    padding = [1, 1, 1, 1]


def train_carracing(args):
    cfg = Config()

    device = torch.device(cfg.device)

    # Create box2d vector env environment wrapper.
    env = DictGymWrapper(
        AsyncVectorEnv(
            [make_vision_box2d_env(cfg.env_name, rank) for rank in range(cfg.n_env)]
        )
    )
    num_actions = env.dict_space.actions.shape[0]

    # Build the networks.
    shared_conv_enc = Conv2dEncoder(
        input_shape=cfg.input_shape,
        channels=cfg.channels,
        kernels=cfg.kernels,
        strides=cfg.strides,
        padding=cfg.padding,
    )
    flat_enc_out_size = shared_conv_enc.get_encoder_output_size(flatten=True)

    flat_shared_conv_enc = nn.Sequential(shared_conv_enc, nn.Flatten())

    q1 = SharedEncoderActionValueNet(
        flat_shared_conv_enc, flat_enc_out_size, num_actions, cfg.hidden_dims
    )
    q2 = SharedEncoderActionValueNet(
        flat_shared_conv_enc, flat_enc_out_size, num_actions, cfg.hidden_dims
    )

    policy = Policy(
        flat_shared_conv_enc, flat_enc_out_size, num_actions, cfg.hidden_dims
    )

    ln_alpha = torch.tensor(np.log(cfg.init_alpha), requires_grad=True, device=device)

    q1 = q1.to(device)
    q2 = q2.to(device)
    policy = policy.to(device)

    # Create the loss callbacks.
    logged_cbs = [
        QLoss(
            name="q1",
            q=q1,
            opt=Adam(q1.parameters(), lr=cfg.learning_rate),
            max_grad_norm=cfg.max_grad_norm,
        ),
        QLoss(
            name="q2",
            q=q2,
            opt=Adam(q2.parameters(), lr=cfg.learning_rate),
            max_grad_norm=cfg.max_grad_norm,
        ),
        PolicyLoss(
            pi=policy,
            ln_alpha=ln_alpha,
            q=q1,
            opt=Adam(policy.non_shared_parameters(), lr=cfg.learning_rate),
            max_grad_norm=cfg.max_grad_norm,
        ),
        AlphaLoss(
            pi=policy,
            ln_alpha=ln_alpha,
            opt=Adam([ln_alpha], lr=cfg.learning_rate),
            n_actions=num_actions,
            max_grad_norm=cfg.max_grad_norm,
            max_alpha=cfg.max_alpha,
        ),
        QTarget(
            pi=policy,
            ln_alpha=ln_alpha,
            q1=q1,
            q2=q2,
            roll_length=cfg.rollout_len,
            reward_scale=0.1,
        ),
        ImageLoggerCallback(),
    ]

    memory_table = DictObsNStepTable(
        spaces=env.dict_space,
        use_terminal_column=False,
        maxlen=cfg.max_memory_size,
        device=device,
    )

    # Add a gym collector callback
    logged_cbs.append(
        ThreadedGymCollector(
            env,
            VisionAgentProxy(policy, device=device),
            TableMemoryProxy(memory_table, use_terminal=False),
            warmup_steps=cfg.batch_size * 3,
            render=False,
        ),
    )

    # Add a tensorboard logger callback.
    callbacks = logged_cbs + [
        TensorboardLogger(
            logged_cbs,
            SummaryWriter(
                log_dir=args.log_dir + "/" + args.name + "_{}".format(time.time())
            ),
            100,
        ),
    ]

    # Create the memory loader and then train.
    dataloader = MemoryLoader(
        memory_table, cfg.batch_size // cfg.rollout_len, cfg.rollout_len, "batch_size"
    )
    trainer = Trainer(callbacks, dataloader)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="carracing-v1")
    parser.add_argument("--log_dir", type=str, default="/mnt/mllogs/emote/carracing")
    args = parser.parse_args()

    train_carracing(args)
