from __future__ import annotations

import pytest

from gymnasium.vector import AsyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

from emote.memory import LoggingProxyWrapper, TableMemoryProxy
from emote.memory.builder import DictObsTable
from emote.typing import DictObservation, DictResponse, EpisodeState, MetaData

from .gym import DictGymWrapper, HitTheMiddle


@pytest.fixture
def table_proxy():
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(spaces=env.dict_space, maxlen=1000, device="cpu")
    return TableMemoryProxy(table, 0, False)


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
        {0: DictResponse({"actions": [0.0]}, {})},
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
            {0: DictResponse({"actions": [0.0]}, {})},
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
            {0: DictResponse({"actions": [0.0]}, {})} if s < 9 else {},
        )

        state = EpisodeState.RUNNING if s < 8 else EpisodeState.TERMINAL

    assert proxy.completed_episodes == 1

def test_report(table_proxy, tmpdir):
    proxy = LoggingProxyWrapper(
        table_proxy,
        SummaryWriter(
            log_dir=tmpdir,
        ),
        2,
    )
    
    agent_metrics = [{"histogram:ones": [1, 1, 1, 1, 1]}, {"one": 1}]
    metrics = {"agent_metrics": agent_metrics, "two": 2}
    metrics_lists = {"three": [3, 3]}

    proxy.report(metrics, metrics_lists)

    for _ in range(2):
        proxy.report({"agent_metrics": [{"histogram:twos": 2}]}, {})
    
    for _ in range(2):
        proxy.report({"one": 1}, {})

    assert "ones" in proxy.hist_logs
    assert "one" in proxy.windowed_scalar and "one" in proxy.windowed_scalar_cumulative
    assert proxy.windowed_scalar_cumulative["one"] == 3
    assert "two" in proxy.windowed_scalar
    assert "three" in proxy.windowed_scalar and "three" in proxy.windowed_scalar_cumulative
    assert proxy.windowed_scalar_cumulative["three"] == 6
    assert "twos" in proxy.hist_logs and len(proxy.hist_logs["twos"]) == 2


def test_get_report(table_proxy, tmpdir):
    proxy = LoggingProxyWrapper(
        table_proxy,
        SummaryWriter(
            log_dir=tmpdir,
        ),
        2,
    )
    
    agent_metrics = [{"histogram:ones": [1, 1, 1, 1, 1]}, {"one": 1}]
    metrics = {"agent_metrics": agent_metrics, "two": 2}
    metrics_lists = {"three": [3, 3]}

    proxy.report(metrics, metrics_lists)
    
    for _ in range(2):
        proxy.report({"one": 1}, {})

    keys = ["histogram:ones", "one", "two", "three", "random"]
    out, out_lists = proxy.get_report(keys)
    for key in keys[:-1]:
        if "histogram" in key:
            assert key in out and key not in out_lists
        else:
            assert key in out and key in out_lists
    
    assert "random" not in out and "random" not in out_lists
    assert out["histogram:ones"] == 1
    assert out["one"] == 1 and out["one/cumulative"] == 3
    assert out_lists["three"] == [3, 3]
