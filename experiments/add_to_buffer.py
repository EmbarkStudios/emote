import numpy as np
import torch
from emote.memory.builder import DictObsNStepTable
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


vae_space = MDPSpace(
    rewards=BoxSpace(dtype=np.float32, shape=(1,)),
    actions=BoxSpace(dtype=np.float32, shape=(28,)),
    state=DictSpace({"features": BoxSpace(dtype=np.float32, shape=tuple([272,]))}),
)
table = DictObsNStepTable(
    spaces=vae_space,
    use_terminal_column=False,
    maxlen=1000,
    device=torch.device('cpu'),
)
seq = {
    'features': [np.ones(272) for _ in range(6)],
    'actions': [np.ones(28) for _ in range(5)],
    'rewards': [[0] for _ in range(5)]
}
table.add_sequence(0, seq)
table.store("memory_output")

"""
proxy = TableMemoryProxy(table=table, use_terminal=False)
action = DictResponse(
    list_data={"actions": np.ones(28)},
    scalar_data={}
)
obs = DictObservation(
    rewards={"rewards": [0]},
    episode_state=EpisodeState.INITIAL,
    array_data={"features": np.ones(272)}
)
agent_id = 0
proxy.add(
    observations={agent_id: obs},
    responses={agent_id: action}
)
for i in range(5):
    obs = DictObservation(
        rewards={"reward": [0]},
        episode_state=EpisodeState.RUNNING,
        array_data={"features": np.ones(272), "rewards": [0]}

    )
    proxy.add(
        observations={agent_id: obs},
        responses={agent_id: action}
    )
obs = DictObservation(
    rewards={"reward": [0]},
    episode_state=EpisodeState.TERMINAL,
    array_data={"features": np.ones(272), "rewards": [0]}
)
proxy.add(
    observations={agent_id: obs},
    responses={agent_id: action}
)
"""
