from os.path import join
from typing import Generator

import pytest
import torch

from gymnasium.vector import AsyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

from emote import Trainer
from emote.callbacks.checkpointing import Checkpointer, CheckpointLoader
from emote.memory import LoggingProxyWrapper, MemoryTableProxy
from emote.memory.builder import DictObsMemoryTable
from emote.trainer import TrainingShutdownException
from emote.typing import DictObservation, DictResponse, EpisodeState, MetaData

from .gym import DictGymWrapper, HitTheMiddle


@pytest.fixture
def table_proxy():
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsMemoryTable(spaces=env.dict_space, maxlen=1000, device="cpu")
    return MemoryTableProxy(table, 0, False)


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

    for idx in range(10):
        proxy.add(
            {
                0: DictObservation(
                    episode_state=EpisodeState.INITIAL if idx == 0 else EpisodeState.RUNNING,
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

    metrics = {"one": 1}
    metrics_lists = {"three": [3, 3], "histogram:ones": [1, 1, 1, 1, 1]}

    proxy.report(metrics, metrics_lists)

    for _ in range(2):
        proxy.report({"histogram:twos": 2}, {})

    for _ in range(2):
        proxy.report({"one": 1}, {})

    assert "ones" in proxy.hist_logs
    assert "one" in proxy.windowed_scalar and "one" in proxy.windowed_scalar_cumulative
    assert proxy.windowed_scalar_cumulative["one"] == 3
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

    metrics = {"one": 1}
    metrics_lists = {"three": [3, 3], "histogram:ones": [1, 1, 1, 1, 1]}

    proxy.report(metrics, metrics_lists)

    for _ in range(2):
        proxy.report({"one": 1}, {})

    keys = ["histogram:ones", "one", "three", "random"]
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


def test_end_cycle(table_proxy, tmpdir):
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

    proxy._end_cycle()


def test_checkpoints(table_proxy, tmpdir):
    def random_onestep_dataloader() -> Generator:
        yield {
            "default": {
                "observation": {"obs": torch.rand(3, 2)},
                "actions": torch.rand(3, 1),
                "q_target": torch.ones(3, 1),
            },
        }
        raise TrainingShutdownException()

    def nostep_dataloader() -> Generator:
        raise TrainingShutdownException()
        yield {}  # Needed to make this a generator.

    proxy1 = LoggingProxyWrapper(
        table_proxy,
        SummaryWriter(
            log_dir=tmpdir,
        ),
        2,
    )

    state = EpisodeState.INITIAL
    for s in range(10):
        proxy1.add(
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

    assert proxy1.completed_episodes == 1

    run_root = join(tmpdir, "chkpt")
    c1 = [
        Checkpointer(restorees=[proxy1], run_root=run_root, checkpoint_interval=1),
    ]

    t1 = Trainer(c1, random_onestep_dataloader())
    t1.state["inf_step"] = 0
    t1.state["bp_step"] = 0
    t1.state["batch_size"] = 0
    t1.train()
    proxy2 = LoggingProxyWrapper(
        table_proxy,
        SummaryWriter(
            log_dir=tmpdir,
        ),
        2,
    )

    assert proxy2.completed_episodes == 0

    c2 = [
        CheckpointLoader(restorees=[proxy2], run_root=run_root, checkpoint_index=0),
    ]
    t2 = Trainer(c2, nostep_dataloader())
    t2.train()

    assert proxy2.completed_episodes == 1
