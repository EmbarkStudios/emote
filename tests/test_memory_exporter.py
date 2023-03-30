import torch
import os
from gymnasium.vector import AsyncVectorEnv
from emote import Trainer
from emote.memory import MemoryLoader, TableMemoryProxy
from emote.memory.callbacks import MemoryExporterCallback, MemoryImporterCallback
from emote.memory.builder import DictObsTable
from emote.nn.gaussian_policy import GaussianMlpPolicy as Policy
from emote.sac import FeatureAgentProxy
from tests.gym import DictGymWrapper, HitTheMiddle, SimpleGymCollector
from emote.callbacks import BackPropStepsTerminator

N_HIDDEN = 10


def test_htm():
    experiment_load_dir = '/home/ali/codes/emote/logs/replay_buffer'

    device = torch.device("cpu")
    env = DictGymWrapper(AsyncVectorEnv(10 * [HitTheMiddle]))
    table = DictObsTable(spaces=env.dict_space, maxlen=10000, device=device)
    memory_proxy = TableMemoryProxy(table)
    dataloader = MemoryLoader(table, 100, 2, "batch_size")
    policy = Policy(2, 1, [N_HIDDEN, N_HIDDEN])
    agent_proxy = FeatureAgentProxy(policy, device)

    exporter = MemoryExporterCallback(
        memory=table,
        target_memory_name='memory',
        inf_steps_per_memory_export=201,
        experiment_root_path=experiment_load_dir,
    )

    callbacks = [
        SimpleGymCollector(
            env, agent_proxy, memory_proxy, warmup_steps=500, render=False
        ),
        exporter,
        BackPropStepsTerminator(500)
    ]

    trainer = Trainer(callbacks, dataloader)
    trainer.train()

    print(exporter.memory.size())


    importer = MemoryImporterCallback(
        memory=DictObsTable(spaces=env.dict_space, maxlen=10000, device=device),
        target_memory_name='memory',
        experiment_load_dir=experiment_load_dir,
    )
    print(importer.memory.size())
    importer.memory.restore(os.path.join(experiment_load_dir, "memory_export"))
    print(importer.memory.size())

    for column in importer.memory._columns.values():
        print('*********************')
        print(importer.memory._columns)
        print(f"data column: {column.name}")
        print("importer keys\n", importer.memory._data[column.name].keys())
        print("\n\n\nexporter keys\n", exporter.memory._data[column.name].keys())
        # print(exporter.memory._data[column.name].values())
        break
        # print(importer.memory._data[column.name] - exporter.memory._data[column.name])


    env.close()


if __name__ == "__main__":
    test_htm()
