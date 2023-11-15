import random
import string

import numpy as np
import pytest
import torch

from emote.algorithms.sac import AlphaLoss, FeatureAgentProxy
from emote.extra.schedules import ConstantSchedule, CyclicSchedule
from emote.nn.gaussian_policy import GaussianMlpPolicy
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


def test_alpha_value_ref_valid_after_load():
    policy = GaussianMlpPolicy(IN_DIM, OUT_DIM, [16, 16])
    init_ln_alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    optim = torch.optim.Adam([init_ln_alpha])
    loss = AlphaLoss(pi=policy, ln_alpha=init_ln_alpha, opt=optim, n_actions=OUT_DIM)

    dummy_load_ln_alpha = torch.tensor(1337.0, dtype=torch.float32, requires_grad=True)
    state_dict = {"network_state_dict": dummy_load_ln_alpha}

    ln_alpha_before_load = loss.ln_alpha
    loss.load_state_dict(state_dict, load_weights=True, load_optimizer=False, load_hparams=False)
    ln_alpha_after_load = loss.ln_alpha

    assert torch.equal(
        ln_alpha_after_load, dummy_load_ln_alpha
    ), "expected to actually load a alpha value."
    assert (
        ln_alpha_before_load is ln_alpha_after_load
    ), "expected ln(alpha) to be the same python object after loading. The reference is used by other loss functions such as PolicyLoss!"


def test_target_entropy_schedules():
    policy = GaussianMlpPolicy(IN_DIM, OUT_DIM, [16, 16])
    init_ln_alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    optim = torch.optim.Adam([init_ln_alpha])
    loss = AlphaLoss(pi=policy, ln_alpha=init_ln_alpha, opt=optim, n_actions=OUT_DIM)

    # Check if default is set correctly when no t_entropy is passed
    init_entropy = loss.t_entropy.value
    assert init_entropy == -OUT_DIM
    print(init_entropy)

    # Check that default schedule is constant and doesn't update the value
    assert isinstance(loss.t_entropy, ConstantSchedule)
    for _ in range(5):
        loss.end_batch()
    assert init_entropy == loss.t_entropy.value

    # Check that value is updated when using a schedule
    start = 5
    end = 0
    steps = 5
    schedule = CyclicSchedule(start, end, steps, mode="triangular")
    loss = AlphaLoss(
        pi=policy, ln_alpha=init_ln_alpha, opt=optim, n_actions=OUT_DIM, t_entropy=schedule
    )

    for _ in range(steps + 1):
        loss.end_batch()
    assert loss.t_entropy.value == end

    for _ in range(steps):
        loss.end_batch()
    assert loss.t_entropy.value == start

    # Check that invalid types are not accepted
    invalid_t_entropy = torch.optim.lr_scheduler.LinearLR(optim, 1, end / start, steps)
    with pytest.raises(TypeError):
        AlphaLoss(
            pi=policy,
            ln_alpha=init_ln_alpha,
            opt=optim,
            n_actions=OUT_DIM,
            t_entropy=invalid_t_entropy,
        )
