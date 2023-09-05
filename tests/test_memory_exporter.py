import os
import stat

import pytest
import torch

from gymnasium.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper, HitTheMiddle, SimpleGymCollector

from emote import Trainer
from emote.callbacks import BackPropStepsTerminator
from emote.memory import MemoryExporterProxyWrapper, MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsTable
from emote.memory.callbacks import MemoryImporterCallback
from emote.memory.storage import BaseStorage
from emote.nn.gaussian_policy import GaussianMlpPolicy as Policy
from emote.sac import FeatureAgentProxy


N_HIDDEN = 10


@pytest.mark.filterwarnings("ignore:Exporting a memory")
def test_memory_export(tmpdir):
    device = torch.device("cpu")
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(spaces=env.dict_space, maxlen=10000, device=device)
    memory_proxy = TableMemoryProxy(table)
    memory_proxy = MemoryExporterProxyWrapper(
        memory=memory_proxy,
        target_memory_name="memory",
        inf_steps_per_memory_export=10,
        experiment_root_path=tmpdir,
        min_time_per_export=1,
    )
    dataloader = MemoryLoader(table, 100, 2, "batch_size")
    policy = Policy(2, 1, [N_HIDDEN, N_HIDDEN])
    agent_proxy = FeatureAgentProxy(policy, device)

    callbacks = [
        SimpleGymCollector(
            env, agent_proxy, memory_proxy, warmup_steps=500, render=False
        ),
        BackPropStepsTerminator(2500),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    importer = MemoryImporterCallback(
        memory=DictObsTable(spaces=env.dict_space, maxlen=10000, device=device),
        target_memory_name="memory",
        experiment_load_dir=tmpdir,
    )

    importer.memory.restore(os.path.join(tmpdir, "memory_export"))

    for column in importer.memory._columns.values():
        if isinstance(importer.memory._data[column.name], BaseStorage):
            for key in importer.memory._data[column.name]:
                assert (
                    importer.memory._data[column.name][key].all()
                    == memory_proxy._inner._table._data[column.name][key].all()
                )

    env.close()


@pytest.mark.filterwarnings("ignore:Exporting a memory")
def test_memory_export_permissions(tmpdir):
    device = torch.device("cpu")
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(spaces=env.dict_space, maxlen=10000, device=device)
    memory_proxy = TableMemoryProxy(table)
    memory_proxy = MemoryExporterProxyWrapper(
        memory=memory_proxy,
        target_memory_name="memory",
        inf_steps_per_memory_export=10,
        experiment_root_path=tmpdir,
        min_time_per_export=1,
    )
    dataloader = MemoryLoader(table, 100, 2, "batch_size")
    policy = Policy(2, 1, [N_HIDDEN, N_HIDDEN])
    agent_proxy = FeatureAgentProxy(policy, device)

    callbacks = [
        SimpleGymCollector(
            env, agent_proxy, memory_proxy, warmup_steps=500, render=False
        ),
        BackPropStepsTerminator(2500),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    assert os.path.exists(os.path.join(tmpdir, "memory_export.zip"))
    st = os.stat(os.path.join(tmpdir, "memory_export.zip"))
    # has to be readable by the whole world
    required_perms = stat.S_IRUSR | stat.S_IROTH | stat.S_IRGRP
    assert (
        st.st_mode & required_perms
    ) == required_perms, "file should be readable by everyone"
