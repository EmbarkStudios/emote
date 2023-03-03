import random
import string

import numpy as np
import pytest
import torch

from emote.nn.gaussian_policy import GaussianMlpPolicy
from emote.sac import FeatureAgentProxy
from emote.typing import DictObservation, EpisodeState


IN_DIM: int = 3
OUT_DIM: int = 2


@pytest.fixture
def random_key():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


@pytest.fixture
def feature_proxy(random_key):
    policy = GaussianMlpPolicy(IN_DIM, OUT_DIM, [16, 16])
    return (
        FeatureAgentProxy(policy, torch.device("cpu"), input_key=random_key),
        random_key,
    )


def test_input_key(feature_proxy):
    agent_proxy, key = feature_proxy
    assert agent_proxy.input_names == (key,), "wrong input key"


def test_input_call(feature_proxy):
    agent_proxy, key = feature_proxy
    result = agent_proxy(
        {
            i: DictObservation(
                rewards={},
                episode_state=EpisodeState.RUNNING,
                array_data={key: np.array([0.0] * IN_DIM, dtype=np.float32)},
            )
            for i in range(3)
        }
    )
    assert len(result) == 3
