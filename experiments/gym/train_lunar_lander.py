import argparse
import time
from functools import partial
import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper
from tests.gym.collector import ThreadedGymCollector
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from emote import Trainer
from emote.callbacks import TensorboardLogger, BackPropStepsTerminator
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


def create_memory(
        env: DictGymWrapper,
        memory_size: int,
        len_rollout: int,
        batch_size: int,
        data_group: str,
        device: torch.device):
    """Creates memory and data_loader for training"""
    table = DictObsNStepTable(
        spaces=env.dict_space,
        use_terminal_column=False,
        maxlen=memory_size,
        device=device,
    )
    memory_proxy = TableMemoryProxy(
        table=table,
        use_terminal=False
    )
    data_loader = MemoryLoader(
        table=table,
        rollout_count=batch_size // len_rollout,
        rollout_length=len_rollout,
        size_key="batch_size",
        data_group=data_group,
    )
    return memory_proxy, data_loader


def create_actor_critic_agents(args, num_obs, num_actions):
    device = args.device
    hidden_dims = [args.hidden_layer_size, args.hidden_layer_size]
    q1 = QNet(num_obs, num_actions, hidden_dims)
    q2 = QNet(num_obs, num_actions, hidden_dims)
    policy = Policy(num_obs, num_actions, hidden_dims)
    q1 = q1.to(device)
    q2 = q2.to(device)
    policy = policy.to(device)
    policy_proxy = FeatureAgentProxy(policy, device=device)
    return q1, q2, policy_proxy


def create_train_callbacks(args, q1, q2, policy_proxy, env, memory_proxy):
    device = torch.device(args.device)
    batch_size = args.batch_size
    max_grad_norm = 1
    len_rollout = args.rollout_length
    num_actions = env.dict_space.actions.shape[0]

    init_alpha = 1.0
    ln_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=device)

    training_cbs = [
        QLoss(
            name="q1",
            q=q1,
            opt=Adam(q1.parameters(), lr=args.critic_lr),
            max_grad_norm=max_grad_norm,
            data_group=args.data_group,
        ),
        QLoss(
            name="q2",
            q=q2,
            opt=Adam(q2.parameters(), lr=args.critic_lr),
            max_grad_norm=max_grad_norm,
            data_group=args.data_group,
        ),
        PolicyLoss(
            pi=policy_proxy.policy,
            ln_alpha=ln_alpha,
            q=q1,
            opt=Adam(policy_proxy.policy.parameters(), lr=args.actor_lr),
            max_grad_norm=max_grad_norm,
            data_group=args.data_group,
        ),
        AlphaLoss(
            pi=policy_proxy.policy,
            ln_alpha=ln_alpha,
            opt=Adam([ln_alpha], lr=args.actor_lr),
            n_actions=num_actions,
            max_grad_norm=max_grad_norm,
            max_alpha=10.0,
            data_group=args.data_group,
        ),
        QTarget(
            pi=policy_proxy.policy,
            ln_alpha=ln_alpha,
            q1=q1,
            q2=q2,
            roll_length=len_rollout,
            reward_scale=0.1,
            data_group=args.data_group,
        ),
        ThreadedGymCollector(
            env,
            policy_proxy,
            memory_proxy,
            warmup_steps=batch_size,
            render=False,
        ),
    ]
    return training_cbs


def create_full_callbacks(args, train_cbs):
    callbacks = train_cbs + [
        TensorboardLogger(
            train_callbacks,
            SummaryWriter(
                log_dir=args.log_dir + "/" + args.name + "_{}".format(time.time())
            ),
            100,
        ),
        BackPropStepsTerminator(
            bp_steps=args.bp_steps,
        ),
    ]
    return callbacks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ll")
    parser.add_argument("--log-dir", type=str, default="/mnt/mllogs/emote/lunar_lander")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--hidden-layer-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=8e-3, help='The policy learning rate')
    parser.add_argument("--critic-lr", type=float, default=8e-3, help='Q-function learning rate')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-group", type=str, default="default")
    parser.add_argument("--bp-steps", type=int, default=10000)

    input_args = parser.parse_args()

    training_device = torch.device(input_args.device)

    gym_wrapper = DictGymWrapper(AsyncVectorEnv([_make_env() for _ in range(input_args.num_envs)]))
    number_of_actions = gym_wrapper.dict_space.actions.shape[0]
    number_of_obs = list(gym_wrapper.dict_space.state.spaces.values())[0].shape[0]

    gym_memory_proxy, dataloader = create_memory(env=gym_wrapper,
                                                 memory_size=4_000_000,
                                                 len_rollout=input_args.len_rollout,
                                                 batch_size=input_args.batch_size,
                                                 data_group='default',
                                                 device=training_device)

    qnet1, qnet2, agent_proxy = create_actor_critic_agents(input_args, number_of_actions, number_of_obs)

    train_callbacks = create_train_callbacks(input_args, qnet1, qnet2, agent_proxy, gym_wrapper, gym_memory_proxy)

    callbacks = create_full_callbacks(input_args, train_callbacks)

    trainer = Trainer(callbacks, dataloader)
    trainer.train()
