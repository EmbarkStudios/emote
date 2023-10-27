"""Sequence builder collates observations into sequences stored in the memory.

The sequence builder is the API between "instant" based APIs such as the agent
proxy and the episode-based functionality of the memory implementation. The goal
of the sequence builder is to consume individual timesteps per agent and collate
them into episodes before submission into the memory.
"""
from __future__ import annotations

import inspect
import logging
import os
import time
import warnings

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from emote.callback import Callback
from emote.extra.onnx_exporter import OnnxExporter
from emote.mixins.logging import LoggingMixin
from emote.proxies import MemoryProxy
from emote.trainer import TrainingShutdownException

from ..typing import AgentId, DictObservation, DictResponse, EpisodeState
from ..utils import BlockTimers, TimedBlock
from .core_types import Matrix
from .table import Table


@dataclass
class Episode:
    """An episode of data being constructed"""

    data: Dict[str, List[Matrix]] = field(default_factory=lambda: defaultdict(list))

    def append(self, observation: Mapping[str, Matrix]) -> Tuple:
        for k, v in observation.items():
            self.data[k].append(v)

    def complete(self, observation: Mapping[str, Matrix]) -> Mapping[str, Matrix]:
        self.append(observation)
        return self.data

    @staticmethod
    def from_initial(observation: Mapping[str, Matrix]) -> "Episode":
        episode = Episode()
        episode.append(observation)
        return episode


################################################################################


class TableMemoryProxy:
    """The sequence builder wraps a sequence-based memory to build full episodes
    from [identity, observation] data. Not thread safe.
    """

    def __init__(
        self,
        table: Table,
        minimum_length_threshold: Optional[int] = None,
        use_terminal: bool = False,
    ):
        self._store: Dict[AgentId, Episode] = {}
        self._table = table
        if minimum_length_threshold is None:
            self._min_length_filter = lambda _: True
        else:
            key = table._length_key
            self._min_length_filter = lambda ep: len(ep[key]) >= minimum_length_threshold

        self._completed_episodes: set[AgentId] = set()
        self._term_states = [EpisodeState.TERMINAL, EpisodeState.INTERRUPTED]
        self._use_terminal = use_terminal

    def size(self):
        return self._table.size()

    def resize(self, new_size: int):
        self._table.resize(new_size)

    def store(self, path: str):
        return self._table.store(path)

    def is_initial(self, identity: int):
        """Returns true if identity is not already used in a partial sequence. Does not
        validate if the identity is associated with a complete episode."""
        return identity not in self._store

    def add(
        self,
        observations: Dict[AgentId, DictObservation],
        responses: Dict[AgentId, DictResponse],
    ):
        completed_episodes = {}
        for agent_id, observation in observations.items():
            data = {space: feature for space, feature in observation.array_data.items()}
            if observation.episode_state != EpisodeState.INITIAL:
                data["rewards"] = observation.rewards["reward"]

            else:
                assert (
                    agent_id not in self._store
                ), f"Agent {agent_id} already has an ongoing episode"

            if observation.episode_state in self._term_states:
                if self._use_terminal:
                    # The terminal value assigned here is the terminal _mask_ value,
                    # not whether it is terminal. In this case, our legacy code
                    # treated all terminals as fatal, i.e., truncated bootstrap.
                    # Since this is the terminal mask value, an interrupted
                    # episode should be 1.0 or "infinite bootstrap horizon"
                    data["terminal"] = float(observation.episode_state == EpisodeState.INTERRUPTED)

                if agent_id not in self._store:
                    # First warn that this is a new agent id:
                    if agent_id in self._completed_episodes:
                        logging.warning("agent_id has already been completed: %d", agent_id)
                    else:
                        logging.warning(
                            "agent_id completed with no previous sequence: %d", agent_id
                        )

                self._completed_episodes.add(agent_id)

                if agent_id not in self._store:
                    # Then continue without sending an empty episode to the table.
                    continue

                ep = self._store.pop(agent_id).complete(data)
                if self._min_length_filter(ep):  # else discard
                    completed_episodes[agent_id] = ep

            else:
                assert agent_id in responses, "Mismatch between observations and responses!"
                response = responses[agent_id]
                data.update(response.list_data)
                data.update(response.scalar_data)

                if agent_id not in self._store:
                    self._store[agent_id] = Episode.from_initial(data)

                else:
                    self._store[agent_id].append(data)

        for agent_id, sequence in completed_episodes.items():
            self._table.add_sequence(agent_id, sequence)

    def timers(self):
        return self._table._timers


class MemoryProxyWrapper:
    """Base class for memory proxy wrappers.
    This class forwards non-existing method accessess to the inner
    MemoryProxy or MemoryProxyWrapper.
    """

    def __init__(self, inner: "MemoryProxyWrapper" | MemoryProxy, **kwargs):
        super().__init__(**kwargs)
        self._inner = inner

    def __getattr__(self, name):
        # get the attribute from inner.
        # if it does not exist, exception will be raised.
        attr = getattr(self._inner, name)

        # for some safety, make sure it is an method.
        # we only want the memory proxy wrapper to forward methods.
        if not inspect.ismethod(attr):
            # NOTE: In python >= 3.10 we should specify
            # 'obj' and 'name' on the AttributeError so Python can provide hints to the user.
            raise AttributeError(
                f"Accessing non-method inner attribute {name} is not allowed.",
            )

        return attr


class TableMemoryProxyWrapper(MemoryProxyWrapper):
    def __init__(self, *, inner: TableMemoryProxy, **kwargs):
        super().__init__(inner=inner, **kwargs)

    def store(self, path: str):
        return self._inner.store(path)


class LoggingProxyWrapper(TableMemoryProxyWrapper, LoggingMixin):
    def __init__(
        self,
        inner: TableMemoryProxy,
        writer: SummaryWriter,
        log_interval: int,
    ):
        super().__init__(inner=inner, default_window_length=1000)

        self.completed_inferences = 0
        self.completed_episodes = 0

        self._writer = writer
        self._log_interval = log_interval
        self._counter = 0
        self._start_time = time.monotonic()
        self._cycle_start_infs = self.completed_inferences
        self._cycle_start_time = time.perf_counter()

        self._infs_at_start = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "completed_inferences": self.completed_inferences,
            "completed_episodes": self.completed_episodes,
            "inference_steps": self._total_infs,
        }

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        load_network: bool = True,
        load_optimizer: bool = True,
        load_hparams: bool = True,
    ) -> dict[str, Any]:
        if load_hparams:
            self.completed_inferences = state_dict.get(
                "completed_inferences", self.completed_inferences
            )
            self.completed_episodes = state_dict.get("completed_episodes", self.completed_episodes)
            self._total_infs = state_dict.get("inference_steps", self._total_infs)
            self._infs_at_start = self.completed_inferences

    def add(
        self,
        observations: Dict[AgentId, DictObservation],
        responses: Dict[AgentId, DictResponse],
    ):
        self._counter += 1

        self.completed_inferences += len(observations)
        self.completed_episodes += len(observations) - len(responses)

        for obs in observations.values():
            if obs.metadata is None:
                continue

            self.report(obs.metadata.info, obs.metadata.info_lists)

        if (self._counter % self._log_interval) == 0:
            self._end_cycle()
            self._counter = 0

        return self._inner.add(observations, responses)

    def report(
        self,
        metrics: dict[str, float],
        metrics_lists: dict[str, list[float]],
    ):
        for key, value in metrics.items():
            if key.startswith("histogram:"):
                self.log_histogram(key[10:], value)
            else:
                self.log_windowed_scalar(key, value)

        for key, value in metrics_lists.items():
            if key.startswith("histogram:"):
                self.log_histogram(key[10:], value)
            else:
                self.log_windowed_scalar(key, value)

    def get_report(
        self, keys: List[str]
    ) -> Tuple[dict[str, Union[int, float, list[float]]], dict[str, list[float]]]:
        keys = set(keys)
        out = {}
        out_lists = {}

        for key in keys:
            if key.startswith("histogram:") and key[10:] in self.hist_logs:
                window = self.hist_logs[key[10:]]
                out[key] = sum(window) / len(window)
            elif key in self.windowed_scalar:
                window = self.windowed_scalar[key]
                out_lists[key] = list(window)
                out[key] = sum(window) / len(window)
                out[f"{key}/cumulative"] = self.windowed_scalar_cumulative[key]

        return out, out_lists

    def _end_cycle(self):
        now_time = time.perf_counter()
        cycle_time = now_time - self._cycle_start_time
        cycle_infs = self.completed_inferences - self._cycle_start_infs
        inf_step = self.completed_inferences
        self.log_scalar("training/inf_per_sec", cycle_infs / cycle_time)
        self.log_scalar("episode/completed", self.completed_episodes)

        for name, (mean, var) in self.timers().stats().items():
            self.log_scalar(f"memory/{self._target_memory_name}/{name}/timing/mean", mean)
            self.log_scalar(f"memory/{self._target_memory_name}/{name}/timing/var", var)

        if "episode/reward" in self.windowed_scalar:
            rewards = self.windowed_scalar["episode/reward"]
            average_reward = sum(rewards) / len(rewards)
            rewards_tensor = torch.Tensor(rewards)

            self._writer.add_scalar(
                "env_vs_episode/reward", average_reward, self.completed_episodes
            )
            self._writer.add_histogram("episode/reward_distribution", rewards_tensor, inf_step)
            self._writer.add_histogram(
                "env_vs_episode/reward_distribution",
                rewards_tensor,
                self.completed_episodes,
            )

        for k, v in self.scalar_logs.items():
            self._writer.add_scalar(k, v, inf_step)

        for k, v in self.windowed_scalar.items():
            k = k.split(":")[1] if k.startswith("windowed[") else k

            self._writer.add_scalar(k, sum(v) / len(v), inf_step)

        for k, v in self.windowed_scalar_cumulative.items():
            k = k.split(":")[1] if k.startswith("windowed[") else k

            self._writer.add_scalar(f"{k}/cumulative", v, inf_step)

        for k, v in self.image_logs.items():
            self._writer.add_image(k, v, inf_step, dataformats="HWC")

        for k, (video_array, fps) in self.video_logs.items():
            self._writer.add_video(k, video_array, inf_step, fps=fps, walltime=None)

        for k, v in self.hist_logs.items():
            if isinstance(v, deque):
                v = np.array(v)

            self._writer.add_histogram(k, v, inf_step)

        time_since_start = time.monotonic() - self._start_time

        self._writer.add_scalar(
            "performance/inf_steps_per_sec",
            (inf_step - self._infs_at_start) / time_since_start,
            inf_step,
        )

        self._writer.flush()

        self._cycle_start_infs = self.completed_inferences
        self._cycle_start_time = now_time


class MemoryExporterProxyWrapper(TableMemoryProxyWrapper, LoggingMixin):
    """Export the memory at regular intervals"""

    def __init__(
        self,
        memory: Union[TableMemoryProxy, TableMemoryProxyWrapper],
        target_memory_name,
        inf_steps_per_memory_export,
        experiment_root_path: str,
        min_time_per_export: int = 600,
    ):
        super().__init__(inner=memory)

        recommended_min_inf_steps = 10_000
        if inf_steps_per_memory_export < recommended_min_inf_steps:
            warnings.warn(
                f"Exporting a memory is a slow operation "
                f"and should not be done too often. "
                f"Current inf_step is {inf_steps_per_memory_export}, "
                f"while the recommended minimum is {recommended_min_inf_steps}.",
                UserWarning,
            )

        self._inf_step = 0
        self.experiment_root_path = experiment_root_path
        self._target_memory_name = target_memory_name
        self._inf_steps_per_memory_export = inf_steps_per_memory_export
        self._min_time_per_export = min_time_per_export

        self._next_export = inf_steps_per_memory_export
        self._next_export_time = time.monotonic() + min_time_per_export
        self._scopes = BlockTimers()

    def add(
        self,
        observations: Dict[AgentId, DictObservation],
        responses: Dict[AgentId, DictResponse],
    ):
        """First add the new batch to the memory"""
        self._inner.add(observations, responses)

        """Save the replay buffer if it has enough data and enough time"""
        has_enough_data = self._inf_step > self._next_export
        time_now = time.monotonic()
        has_enough_time = time_now > self._next_export_time

        self._inf_step += 1

        if has_enough_data and has_enough_time:
            logging.info("Starting Memory export...")
            start_time = time.time()
            self._next_export = self._inf_step + self._inf_steps_per_memory_export
            self._next_export_time = time_now + self._min_time_per_export

            export_path = os.path.join(
                self.experiment_root_path, f"{self._target_memory_name}_export"
            )
            with self._scopes.scope("export"):
                self._inner.store(export_path)

            elapsed_time = time.time() - start_time
            logging.info(f"Memory export completed in {elapsed_time} seconds")

            for name, (mean, var) in self._scopes.stats().items():
                self.log_scalar(f"memory/{self._target_memory_name}/{name}/timing/mean", mean)
                self.log_scalar(f"memory/{self._target_memory_name}/{name}/timing/var", var)


class MemoryLoader:
    def __init__(
        self,
        table: Table,
        rollout_count: int,
        rollout_length: int,
        size_key: str,
        data_group: str = "default",
    ):
        self.data_group = data_group
        self.table = table
        self.rollout_count = rollout_count
        self.rollout_length = rollout_length
        self.size_key = size_key
        self.timer = TimedBlock()

    def is_ready(self):
        """True if the data loader has enough data to start providing data"""
        return self.table.size() >= (self.rollout_count * self.rollout_length)

    def __iter__(self):
        if not self.is_ready():
            raise Exception(
                "Data loader does not have enough data.\
                 Check `is_ready()` before trying to iterate over data."
            )

        while True:
            with self.timer:
                data = self.table.sample(self.rollout_count, self.rollout_length)

            data[self.size_key] = self.rollout_count * self.rollout_length
            yield {self.data_group: data, self.size_key: data[self.size_key]}


class JointMemoryLoader:
    """A memory loader capable of loading data from multiple `MemoryLoader`s.

    If a datagroup is specified via the `data_group` param, it will place all loaded data in the specified datagroup,
    otherwise it will directly return the data fetched from each individual loader.
    """

    def __init__(self, loaders: list[MemoryLoader], data_group: str | None = None):
        self._loaders = loaders
        self._data_group = data_group

    def is_ready(self):
        return all(loader.is_ready() for loader in self._loaders)

    def __iter__(self):
        if not self.is_ready():
            raise Exception(
                """memory loader(s) in JointMemoryLoader does not have enough data. Check `is_ready()`
                before trying to iterate over data."""
            )

        while True:
            out = {}
            for loader in self._loaders:
                out.update(next(iter(loader)))

            if self._data_group is not None:
                out = {self._data_group: out}

            yield out


class MemoryWarmup(Callback):
    """A blocker to ensure memory has data.

    This ensures the memory has enough data when training starts, as the memory
    will panic otherwise. This is useful if you use an async data generator.

    If you do not use an async data generator this can deadlock your training
    loop and prevent progress.
    """

    def __init__(
        self,
        loader: MemoryLoader,
        exporter: Optional[OnnxExporter],
        shutdown_signal: Optional[Callable[[], bool]] = None,
    ):
        super().__init__()
        self._order = 100
        self._loader = loader
        self._exporter = exporter
        self._shutdown_signal = shutdown_signal or (lambda: False)

    def begin_training(self):
        import time

        while not self._loader.is_ready():
            time.sleep(0.1)
            if self._exporter:
                self._exporter.process_pending_exports()

            if self._shutdown_signal():
                raise TrainingShutdownException
