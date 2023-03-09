from __future__ import annotations

import pytest
import torch

from gym.vector import AsyncVectorEnv
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from emote import Trainer
from emote.callback import Callback
from emote.callbacks import FinalRewardTestCheck, TerminalLogger
from emote.extra.onnx_exporter import OnnxExporter
from emote.memory import LoggingProxyWrapper, MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsTable
from emote.nn.gaussian_policy import GaussianMlpPolicy as Policy
from emote.sac import AlphaLoss, FeatureAgentProxy, PolicyLoss, QLoss, QTarget
from emote.trainer import Trainer, TrainingShutdownException
from emote.typing import AgentId, DictObservation, DictResponse, EpisodeState, MetaData

from .gym import DictGymWrapper, HitTheMiddle, SimpleGymCollector


@pytest.fixture
def table_proxy():
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(spaces=env.dict_space, maxlen=1000, device="cpu")
    return TableMemoryProxy(table, 0, True)


def test_construct(table_proxy, tmpdir):
    _ = LoggingProxyWrapper(
        table_proxy,
        SummaryWriter(
            log_dir=tmpdir,
        ),
        2,
    )


def test_add_once(table_proxy, tmpdir):
    proxy = LoggingProxyWrapper(
        table_proxy,
        SummaryWriter(
            log_dir=tmpdir,
        ),
        2,
    )

    proxy.add(
        {
            0: DictObservation(
                episode_state=EpisodeState.INITIAL,
                array_data={"obs": [1.0]},
                rewards={"reward": None},
                metadata=MetaData(info={"episode/reward": 10.0}, info_lists={}),
            )
        },
        {0: DictResponse({"action": [0.0]}, {})},
    )

    assert "episode/reward" in proxy.windowed_scalar


def test_add_multiple(table_proxy, tmpdir):
    proxy = LoggingProxyWrapper(
        table_proxy,
        SummaryWriter(
            log_dir=tmpdir,
        ),
        2,
    )

    for _ in range(10):
        proxy.add(
            {
                0: DictObservation(
                    episode_state=EpisodeState.INITIAL,
                    array_data={"obs": [1.0]},
                    rewards={"reward": None},
                    metadata=MetaData(info={"episode/reward": 10.0}, info_lists={}),
                )
            },
            {0: DictResponse({"action": [0.0]}, {})},
        )

    assert "training/inf_per_sec" in proxy.scalar_logs


def test_completed(table_proxy, tmpdir):
    proxy = LoggingProxyWrapper(
        table_proxy,
        SummaryWriter(
            log_dir=tmpdir,
        ),
        2,
    )

    state = EpisodeState.INITIAL
    for s in range(10):
        proxy.add(
            {
                0: DictObservation(
                    episode_state=state,
                    array_data={"obs": [1.0]},
                    rewards={"reward": None},
                    metadata=MetaData(info={"episode/reward": 10.0}, info_lists={}),
                )
            },
            {0: DictResponse({"action": [0.0]}, {})} if s < 9 else {},
        )

        state = EpisodeState.RUNNING if s < 8 else EpisodeState.TERMINAL

    assert proxy.completed_episodes == 1
