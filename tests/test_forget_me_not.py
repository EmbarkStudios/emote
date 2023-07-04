from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch

from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from emote import Trainer
from emote.callbacks.logging import TensorboardLogger
from emote.callbacks.testing import FinalRewardTestCheck
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsNStepTable
from emote.nn.gaussian_policy import GaussianMlpPolicy as Policy, GaussianPolicyHead
from emote.nn.initialization import ortho_init_, xavier_uniform_init_
from emote.nn.rnn import (
    BurnInCallback,
    BurnInDictObsAdaptor,
    BurnInSamplerAdaptor,
    GruEncoder,
)
from emote.sac import (
    AlphaLoss,
    FeatureAgentProxy,
    GenericAgentProxy,
    PolicyLoss,
    QLoss,
    QTarget,
)

from .gym import DictGymWrapper
from .gym.collector import SimpleGymCollector, SimpleRecurrentGymCollector
from .gym.forget_me_not import ForgetMeNot


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip([input_dim] + hidden_dims, hidden_dims)
            ],
        )

        self.encoder.apply(ortho_init_)

    def forward(self, state: Tensor) -> Tensor:
        return self.encoder(state)


class QGruNet(nn.Module):
    def __init__(
        self,
        obs: int,
        act: int,
        hidden_dims: list[int],
        shared_encoder: GruEncoder,
    ):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.encoder = MLPEncoder(obs + act, hidden_dims)
        self.q_out = nn.Linear(hidden_dims[-1], 1)
        self.q_out.apply(partial(xavier_uniform_init_, gain=0.01))

    def forward(self, action, obs, gru_hidden):
        obs, gru_hidden = self.shared_encoder(obs, gru_hidden)

        x = torch.cat([obs, action], dim=1)
        x = self.encoder(x)
        return self.q_out(x)


class GaussianGruPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        shared_encoder: GruEncoder,
    ):
        super().__init__()
        self.shared_encoder = shared_encoder
        self.encoder = MLPEncoder(observation_dim, hidden_dims)

        self.policy = GaussianPolicyHead(hidden_dims[-1], action_dim)
        self.policy.apply(partial(xavier_uniform_init_, gain=0.01))

    def forward(
        self, obs: Tensor, gru_hidden: Tensor, epsilon: Tensor | None = None
    ) -> Tensor | tuple[Tensor, ...]:
        encoded, gru_hidden = self.shared_encoder(obs, gru_hidden)
        encoded = self.encoder(encoded)
        return (
            *self.policy(encoded, epsilon),
            gru_hidden,
        )


def _thunk(*args, **kwargs):
    return ForgetMeNot(
        *args,
        **kwargs,
    )


@pytest.fixture
def base_config() -> dict[str, any]:
    return {
        "n_hidden": 64,
        "gru_size": 32,
        "obs_size": 3,
        "rollout_length": 3,
        "batch_size": 1000,
        "memory_max_size": 1_000_000,
        "num_envs": 10,
        "warmup_steps": 1000,
        "max_steps": 1_000,
        "learning_rate": 2.5e-3,
        "game_length": 4,
        "min_memory_gap": 1,
        "max_memory_gap": 1,
        "num_actions": 1,
        "use_burn_in": False,
        "burn_in_length": 1,
    }


def _run_gru_test(root_dir, suffix, config):
    device = torch.device("cpu")
    env = DictGymWrapper(
        AsyncVectorEnv(
            10
            * [
                partial(
                    _thunk,
                    num_actions=config["num_actions"],
                    game_length=config["game_length"],
                    min_memory_gap=config["min_memory_gap"],
                    max_memory_gap=config["max_memory_gap"],
                )
            ]
        )
    )
    space = env.dict_space
    space.state.spaces["gru_hidden"] = spaces.Box(
        -np.ones(config["gru_size"]), np.ones(config["gru_size"])
    )
    adaptors = []
    if config["use_burn_in"]:
        adaptors.append(
            BurnInSamplerAdaptor(
                ["obs", "gru_hidden", "actions", "rewards", "masks"],
                ["obs"],
                config["burn_in_length"],
            )
        )

        adaptors.append(
            BurnInDictObsAdaptor(
                ["obs", "gru_hidden"],
            )
        )

    table = DictObsNStepTable(
        spaces=space,
        use_terminal_column=False,
        maxlen=config["memory_max_size"],
        device=device,
        adaptors=adaptors,
    )

    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(
        table, config["batch_size"], config["rollout_length"], size_key="batch_size"
    )

    rollout_length = config["rollout_length"]

    shared_encoder = GruEncoder(
        encoder_dim=config["n_hidden"],
        gru_hidden=config["gru_size"],
        batch_size=config["batch_size"],
        rollout_length=rollout_length,
        input_encoder=MLPEncoder(config["obs_size"], [config["n_hidden"]]),
    )
    q1 = QGruNet(
        config["gru_size"], config["num_actions"], [config["n_hidden"]], shared_encoder
    )
    q2 = QGruNet(
        config["gru_size"], config["num_actions"], [config["n_hidden"]], shared_encoder
    )

    policy = GaussianGruPolicy(
        config["gru_size"], config["num_actions"], [config["n_hidden"]], shared_encoder
    )
    ln_alpha = torch.tensor(0.1, requires_grad=True)
    agent_proxy = GenericAgentProxy(
        policy, device, ("obs", "gru_hidden"), ("actions", "gru_hidden")
    )

    logged_cbs = [
        QLoss(name="q1", q=q1, opt=Adam(q1.parameters(), lr=config["learning_rate"])),
        QLoss(name="q2", q=q2, opt=Adam(q2.parameters(), lr=config["learning_rate"])),
        PolicyLoss(pi=policy, ln_alpha=ln_alpha, q=q1, opt=Adam(policy.parameters())),
        AlphaLoss(
            pi=policy,
            ln_alpha=ln_alpha,
            opt=Adam([ln_alpha]),
            n_actions=config["num_actions"],
        ),
        QTarget(pi=policy, ln_alpha=ln_alpha, q1=q1, q2=q2, roll_length=rollout_length),
        SimpleRecurrentGymCollector(
            env,
            agent_proxy,
            memory_proxy,
            "gru_hidden",
            np.zeros(config["gru_size"], dtype=np.float32),
            warmup_steps=config["warmup_steps"],
            render=False,
        ),
    ]

    if config["use_burn_in"]:
        logged_cbs.insert(
            0,
            BurnInCallback(
                ["obs", "gru_hidden"],
                "gru_hidden",
                shared_encoder,
                config["burn_in_length"],
            ),
        )

    callbacks = logged_cbs + [
        TensorboardLogger(
            logged_cbs,
            SummaryWriter(
                log_dir=root_dir / suffix,
            ),
            100,
        ),
        FinalRewardTestCheck(
            logged_cbs[-1],
            0.6,
            config["max_steps"],
            key="final/smooth_reward",
            use_windowed=True,
        ),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    env.close()
    pass


def test_gru(tmpdir, base_config):
    _run_gru_test(tmpdir, "gru_base", base_config)


def test_gru_long_episodes(tmpdir, base_config):
    base_config["game_length"] = 10
    base_config["max_memory_gap"] = 5
    base_config["max_steps"] = 4_000
    _run_gru_test(tmpdir, "gru_long_episodes", base_config)


def test_gru_long_episodes_burn_in1(tmpdir, base_config):
    base_config["game_length"] = 10
    base_config["max_memory_gap"] = 5
    base_config["max_steps"] = 4_000

    base_config["use_burn_in"] = True
    base_config["burn_in_length"] = 1
    _run_gru_test(tmpdir, "gru_long_episodes_b1", base_config)


def test_gru_long_episodes_burn_in3(tmpdir, base_config):
    base_config["game_length"] = 10
    base_config["max_memory_gap"] = 5
    base_config["max_steps"] = 4_000
    base_config["use_burn_in"] = True
    base_config["burn_in_length"] = 3
    _run_gru_test(tmpdir, "gru_long_episodes_b3", base_config)


def test_gru_long_episodes_burn_in5(tmpdir, base_config):
    base_config["game_length"] = 10
    base_config["max_memory_gap"] = 5
    base_config["max_steps"] = 4_000
    base_config["use_burn_in"] = True
    base_config["burn_in_length"] = 5
    _run_gru_test(tmpdir, "gru_long_episodes_b5", base_config)


class QNet(nn.Module):
    def __init__(self, obs: int, act: int, hidden_dims: list[int]):
        super().__init__()
        self.encoder = MLPEncoder(obs + act, hidden_dims)
        self.q = nn.Linear(hidden_dims[-1], 1)

    def forward(self, action, obs):
        x = torch.cat([obs, action], dim=1)
        x = self.encoder(x)
        return self.q(x)


def _run_mlp_test(root_dir, suffix, config):
    device = torch.device("cpu")
    env = DictGymWrapper(
        AsyncVectorEnv(
            10
            * [
                partial(
                    _thunk,
                    num_actions=config["num_actions"],
                    game_length=config["game_length"],
                    min_memory_gap=config["min_memory_gap"],
                    max_memory_gap=config["max_memory_gap"],
                )
            ]
        )
    )

    table = DictObsNStepTable(
        spaces=env.dict_space,
        use_terminal_column=False,
        maxlen=config["memory_max_size"],
        device=device,
    )
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(
        table, config["batch_size"], config["rollout_length"], "batch_size"
    )

    q1 = QNet(
        config["obs_size"],
        config["num_actions"],
        [config["n_hidden"], config["n_hidden"]],
    )
    q2 = QNet(
        config["obs_size"],
        config["num_actions"],
        [config["n_hidden"], config["n_hidden"]],
    )
    policy = Policy(
        config["obs_size"],
        config["num_actions"],
        [config["n_hidden"], config["n_hidden"]],
    )
    ln_alpha = torch.tensor(0.1, requires_grad=True)
    agent_proxy = FeatureAgentProxy(policy, device)

    logged_cbs = [
        QLoss(name="q1", q=q1, opt=Adam(q1.parameters(), lr=config["learning_rate"])),
        QLoss(name="q2", q=q2, opt=Adam(q2.parameters(), lr=config["learning_rate"])),
        PolicyLoss(pi=policy, ln_alpha=ln_alpha, q=q1, opt=Adam(policy.parameters())),
        AlphaLoss(
            pi=policy,
            ln_alpha=ln_alpha,
            opt=Adam([ln_alpha]),
            n_actions=config["num_actions"],
        ),
        QTarget(
            pi=policy,
            ln_alpha=ln_alpha,
            q1=q1,
            q2=q2,
            roll_length=config["rollout_length"],
        ),
        SimpleGymCollector(
            env,
            agent_proxy,
            memory_proxy,
            warmup_steps=config["warmup_steps"],
            render=False,
        ),
    ]

    callbacks = logged_cbs + [
        TensorboardLogger(
            logged_cbs,
            SummaryWriter(
                log_dir=root_dir / suffix,
            ),
            100,
        ),
        FinalRewardTestCheck(
            logged_cbs[5],
            0.5,
            config["max_steps"],
            key="final/smooth_reward",
            use_windowed=True,
        ),
    ]

    trainer = Trainer(callbacks, dataloader)
    with pytest.raises(Exception, match="Reward too low: [-0-9.]+"):
        trainer.train()

    env.close()


def test_mlp_fails(tmpdir, base_config):
    _run_mlp_test(tmpdir, "mlp_fails", base_config)
