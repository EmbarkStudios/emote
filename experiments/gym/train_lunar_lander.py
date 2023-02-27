import argparse
import time

from functools import partial

import gym
import numpy as np
import torch

from gym.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper
from tests.gym.collector import ThreadedGymCollector
from emote.models.model_env import ModelBasedCollector
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from emote import Trainer
from emote.callbacks import FinalLossTestCheck, TensorboardLogger
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsNStepTable
from emote.nn import GaussianPolicyHead
from emote.nn.initialization import ortho_init_, xavier_uniform_init_
from emote.sac import AlphaLoss, FeatureAgentProxy, PolicyLoss, QLoss, QTarget

from typing import List, Union

from emote.models.ensemble import EnsembleOfGaussian
from emote.models.model import DynamicModel, ModelLoss
from emote.models.model_env import ModelEnv


def _make_env(rank):
    def _thunk():
        env = gym.make("LunarLander-v2", continuous=True)
        env = gym.wrappers.FrameStack(env, 3)
        env = gym.wrappers.FlattenObservation(env)
        #env.seed(rank)
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
    logged_cbs: List[Union[QLoss,
                     PolicyLoss,
                     AlphaLoss,
                     QTarget,
                     ThreadedGymCollector,
                     ModelLoss,
                     ModelBasedCollector,
                     TensorboardLogger]]

    device = torch.device(args.device)

    hidden_dims = [256, 256]
    batch_size = args.batch_size
    n_env = args.num_envs
    max_grad_norm = 1

    rollout_len = args.rollout_length
    init_alpha = 1.0

    env = DictGymWrapper(AsyncVectorEnv([_make_env(i) for i in range(n_env)]))
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
            opt=Adam(q1.parameters(), lr=args.critic_lr),
            max_grad_norm=max_grad_norm,
        ),
        QLoss(
            name="q2",
            q=q2,
            opt=Adam(q2.parameters(), lr=args.critic_lr),
            max_grad_norm=max_grad_norm,
        ),
        PolicyLoss(
            pi=policy,
            ln_alpha=ln_alpha,
            q=q1,
            opt=Adam(policy.parameters(), lr=args.actor_lr),
            max_grad_norm=max_grad_norm,
        ),
        AlphaLoss(
            pi=policy,
            ln_alpha=ln_alpha,
            opt=Adam([ln_alpha], lr=args.actor_lr),
            n_actions=num_actions,
            max_grad_norm=max_grad_norm,
            max_alpha=10.0,
        ),
        QTarget(
            pi=policy,
            ln_alpha=ln_alpha,
            q1=q1,
            q2=q2,
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

    if args.model_based:
        assert rollout_len == 1, "--rollout-length must be set to 1 for model-based training"
        model = EnsembleOfGaussian(num_obs + num_actions, num_obs + 1, device, ensemble_size=args.num_model_ensembles)
        dynamic_model = DynamicModel(model=model)

        init_rollout_length = 1
        init_maxlen = (
                init_rollout_length *
                args.num_model_envs *
                args.num_bp_to_retain_sac_buffer
        )

        sac_buffer = DictObsNStepTable(
            spaces=env.dict_space,
            use_terminal_column=False,
            maxlen=init_maxlen,
            device=device,
        )
        sac_buffer_proxy = TableMemoryProxy(sac_buffer, use_terminal=False)
        model_env = ModelEnv(env=gym.make("LunarLander-v2", continuous=True),
                             num_envs=args.num_model_envs,
                             model=dynamic_model,
                             termination_fn=lambda _, __: torch.tensor([False]),
                             )

        logged_cbs = logged_cbs + [
            ModelLoss(
                model=dynamic_model,
                name='dynamic_model',
                opt=Adam(model.parameters(), lr=args.model_lr),
            ),
            ModelBasedCollector(
                model_env=model_env,
                agent=agent_proxy,
                memory=sac_buffer_proxy,
                rollout_length=1
            )
        ]

    callbacks = logged_cbs + [
        TensorboardLogger(
            logged_cbs,
            SummaryWriter(
                log_dir=args.log_dir + "/" + args.name + "_{}".format(time.time())
            ),
            100,
        ),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ll")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--log-dir", type=str, default="/mnt/mllogs/emote/lunar_lander")
    parser.add_argument('--model-based', action='store_true')
    parser.add_argument("--num-model-ensembles", type=int, default=5,
                        help='The number of dynamic models in the ensemble')
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-bp-to-retain-sac-buffer", type=int, default=5000)
    parser.add_argument("--num-model-envs", type=int, default=50)
    parser.add_argument("--model-lr", type=float, default=1e-3, help='The model learning rate')
    parser.add_argument("--actor-lr", type=float, default=8e-3, help='The policy learning rate')
    parser.add_argument("--critic-lr", type=float, default=8e-3, help='Q-function learning rate')
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    train_lunar_lander(args)
