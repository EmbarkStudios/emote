import argparse
import torch
from torch import nn, optim
from functools import partial
import gym
from gym.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper
from tests.gym.collector import ThreadedGymCollector
from emote import Trainer
from emote.memory.builder import DictObsNStepTable
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.nn import GaussianPolicyHead
from emote.nn.initialization import ortho_init_, xavier_uniform_init_
from emote.sac import AlphaLoss, FeatureAgentProxy, PolicyLoss, QLoss, QTarget
from emote.models.ensemble import EnsembleOfGaussian
from emote.models.model import DynamicModel, ModelLoss


def _make_env(rank):
    def _thunk():
        env = gym.make("BipedalWalker-v3")
        env = gym.wrappers.FrameStack(env, 3)
        env = gym.wrappers.FlattenObservation(env)
        return env

    return _thunk


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
        log_prob = log_prob.clamp(min=-2)
        return sample, log_prob


def train_bipedal_walker(args):
    num_envs = args.num_envs
    env = DictGymWrapper(AsyncVectorEnv([_make_env(i) for i in range(num_envs)]))
    device = torch.device(args.device)
    batch_size = args.batch_size

    table = DictObsNStepTable(
        spaces=env.dict_space,
        use_terminal_column=False,
        maxlen=4_000_000,
        device=device,
    )
    rollout_len = 1
    hidden_dims = [256, 256]
    memory_proxy = TableMemoryProxy(table, use_terminal=False)
    dataloader = MemoryLoader(table, batch_size // rollout_len, rollout_len, "batch_size")

    num_actions = env.dict_space.actions.shape[0]
    num_obs = list(env.dict_space.state.spaces.values())[0].shape[0]

    print('Observation space dim: {:d}, action space dim: {:d}'.format(num_obs, num_actions))

    # q1 = QNet(num_obs, num_actions, hidden_dims).to(device)
    # q2 = QNet(num_obs, num_actions, hidden_dims).to(device)

    policy = Policy(num_obs, num_actions, hidden_dims).to(device)
    agent_proxy = FeatureAgentProxy(policy, device=device)

    model = EnsembleOfGaussian(num_obs+num_actions, num_obs, device, ensemble_size=5)
    dynamic_model = DynamicModel(model=model)

    callbacks = [
        ModelLoss(
            model=dynamic_model,
            name='dynamic_model',
            opt=optim.Adam(model.parameters(), lr=args.model_lr),
            ),
        ThreadedGymCollector(
            env,
            agent_proxy,
            memory_proxy,
            warmup_steps=batch_size,
            render=False,
        ),
    ]
    trainer = Trainer(callbacks, dataloader)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="bipedal_walker")
    parser.add_argument("--log-dir", type=str, default="/mnt/mllogs/emote/bipedal_walker")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--model-lr", type=float, default=1e-3)
    parser.add_argument("--policy-lr", type=float, default=8e-3)
    parser.add_argument("--q-lr", type=float, default=8e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    args_value = parser.parse_args()

    train_bipedal_walker(args_value)
