import os

from tempfile import mkdtemp

import torch

from gymnasium.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper, HitTheMiddle, SimpleGymCollector

from emote import Trainer
from emote.callbacks import BackPropStepsTerminator
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.builder import DictObsTable
from emote.memory.callbacks import MemoryExporterCallback, MemoryImporterCallback
from emote.memory.storage import BaseStorage
from emote.nn.gaussian_policy import GaussianMlpPolicy as Policy
from emote.sac import FeatureAgentProxy


N_HIDDEN = 10


def test_htm_memory_export():
    experiment_load_dir = mkdtemp()

    device = torch.device("cpu")
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(spaces=env.dict_space, maxlen=10000, device=device)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table, 100, 2, "batch_size")
    policy = Policy(2, 1, [N_HIDDEN, N_HIDDEN])
    agent_proxy = FeatureAgentProxy(policy, device)

    exporter = MemoryExporterCallback(
        memory=table,
        target_memory_name="memory",
        inf_steps_per_memory_export=201,
        experiment_root_path=experiment_load_dir,
    )

    callbacks = [
        SimpleGymCollector(
            env, agent_proxy, memory_proxy, warmup_steps=500, render=False
        ),
        exporter,
        BackPropStepsTerminator(500),
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    importer = MemoryImporterCallback(
        memory=DictObsTable(spaces=env.dict_space, maxlen=10000, device=device),
        target_memory_name="memory",
        experiment_load_dir=experiment_load_dir,
    )

    importer.memory.restore(os.path.join(experiment_load_dir, "memory_export"))

    for column in importer.memory._columns.values():
        if isinstance(importer.memory._data[column.name], BaseStorage):
            for key in importer.memory._data[column.name]:
                reverted_key = -(key + 1)
                assert (
                    importer.memory._data[column.name][key].all()
                    == exporter.memory._data[column.name][reverted_key].all()
                )

    env.close()
