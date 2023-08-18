import numpy as np
import torch.types

from tests.test_vae import get_conditioning_fn

from emote.algos.genrl.nn import GenRLPolicy, GenRLQNet
from emote.algos.genrl.proxies import MemoryProxyWithEncoder
from emote.algos.vae.nn import DecoderWrapper, EncoderWrapper
from emote.memory.builder import DictObsTable
from emote.typing import DictObservation, DictResponse, EpisodeState, MetaData
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


LATENT_SIZE = 3
ACTION_SIZE = 2
OBSERVATION_SIZE = 24
CONDITION_SIZE = 24
BATCH_SIZE = 50

HIDDEN_SIZES = [256] * 2


def test_genrl():

    cfn = get_conditioning_fn(CONDITION_SIZE)
    device = torch.device("cpu")

    decoder = DecoderWrapper(
        latent_size=LATENT_SIZE,
        output_size=ACTION_SIZE,
        condition_size=CONDITION_SIZE,
        condition_fn=cfn,
        hidden_sizes=HIDDEN_SIZES,
        device=device,
    )

    encoder = EncoderWrapper(
        latent_size=LATENT_SIZE,
        action_size=ACTION_SIZE,
        hidden_sizes=HIDDEN_SIZES,
        condition_size=CONDITION_SIZE,
        condition_fn=cfn,
        device=device,
    )

    q = GenRLQNet(encoder, OBSERVATION_SIZE, LATENT_SIZE, HIDDEN_SIZES).to(device)
    policy = GenRLPolicy(decoder, OBSERVATION_SIZE, LATENT_SIZE, HIDDEN_SIZES).to(
        device
    )

    obs = torch.rand(BATCH_SIZE, OBSERVATION_SIZE)

    actions, log_prob = policy.forward(obs)
    q_vals = q.forward(actions, obs)

    assert actions.shape == (BATCH_SIZE, ACTION_SIZE)
    assert q_vals.shape == (BATCH_SIZE, 1)
    assert log_prob.shape == (BATCH_SIZE, 1)


def test_memory_proxy():

    cfn = get_conditioning_fn(CONDITION_SIZE)
    device = torch.device("cpu")

    encoder = EncoderWrapper(
        latent_size=LATENT_SIZE,
        action_size=ACTION_SIZE,
        hidden_sizes=HIDDEN_SIZES,
        condition_size=CONDITION_SIZE,
        condition_fn=cfn,
        device=device,
    )

    space = MDPSpace(
        rewards=None,
        actions=BoxSpace(dtype=np.float32, shape=(ACTION_SIZE,)),
        state=DictSpace(
            spaces={"obs": BoxSpace(dtype=np.float32, shape=(OBSERVATION_SIZE,))}
        ),
    )

    table = DictObsTable(spaces=space, maxlen=1000, device=device)

    proxy = MemoryProxyWithEncoder(
        table=table, encoder=encoder, minimum_length_threshold=1, use_terminal=True
    )

    agent_id = 0
    obs = np.random.rand(OBSERVATION_SIZE)
    action = np.random.rand(ACTION_SIZE)

    proxy.add(
        {
            agent_id: DictObservation(
                episode_state=EpisodeState.INITIAL,
                array_data={"obs": obs},
                rewards={"reward": None},
            )
        },
        {0: DictResponse({"actions": action}, {})},
    )

    assert (obs == proxy._store[agent_id].data["obs"][0]).all()
