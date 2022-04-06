# This file has been taken from https://github.com/jackharmer/agency (MIT License)

import torch
from emote.utils.gamma_matrix import make_gamma_matrix, discount
from pytest import approx


def simple_discount(rewards, gamma, value, masks):
    discounts = []
    V = value
    for cc in reversed(range(len(rewards))):
        V = rewards[cc] + gamma * masks[cc] * V
        discounts.append(V)
    return list(reversed(discounts))


def test_simple_discount_works():
    rewards = [0.1, 0.2, 0.3, 0.4]
    masks = [1, 1, 1, 1]
    gamma = 0.9
    value = 10

    # Create the true values
    v3 = rewards[3] + gamma * masks[3] * value
    v2 = rewards[2] + gamma * masks[2] * v3
    v1 = rewards[1] + gamma * masks[1] * v2
    v0 = rewards[0] + gamma * masks[0] * v1

    d_true = [v0, v1, v2, v3]

    d = simple_discount(rewards, gamma, value, masks)

    assert d_true == approx(d, 1e-5)

    masks = [1, 1, 1, 0]

    v3 = rewards[3] + gamma * masks[3] * value
    v2 = rewards[2] + gamma * masks[2] * v3
    v1 = rewards[1] + gamma * masks[1] * v2
    v0 = rewards[0] + gamma * masks[0] * v1

    d_true = [v0, v1, v2, v3]

    d = simple_discount(rewards, gamma, value, masks)

    assert d_true == approx(d, 1e-5)


def test_gamma_matrix():
    rewards = [0.1, 0.2, 0.3, 0.4]
    masks = [1, 1, 1, 1]
    gamma = 0.9
    value = 10
    gamma_matrix = make_gamma_matrix(gamma, len(rewards))

    d_simple = simple_discount(rewards, gamma, value, masks)
    d_gamma = discount(
        torch.tensor(rewards).unsqueeze(0),
        torch.tensor([value * masks[-1]]).unsqueeze(0),
        gamma_matrix,
    )

    assert d_gamma.cpu().numpy() == approx(d_simple, 1e-5)

    masks = [1, 1, 1, 0]
    d_simple = simple_discount(rewards, gamma, value, masks)
    d_gamma = discount(
        torch.tensor(rewards).unsqueeze(0),
        torch.tensor([value * masks[-1]]).unsqueeze(0),
        gamma_matrix,
    )

    assert d_gamma.cpu().numpy() == approx(d_simple, 1e-5)


def test_gamma_matrix_roll1():
    rewards = [0.1]
    masks = [1]
    gamma = 0.9
    value = 10
    gamma_matrix = make_gamma_matrix(gamma, len(rewards))

    d_simple = simple_discount(rewards, gamma, value, masks)
    d_gamma = discount(
        torch.tensor(rewards).unsqueeze(0),
        torch.tensor([value * masks[-1]]).unsqueeze(0),
        gamma_matrix,
    )

    assert d_gamma.cpu().numpy() == approx(d_simple, 1e-5)

    masks = [0]
    d_simple = simple_discount(rewards, gamma, value, masks)
    d_gamma = discount(
        torch.tensor(rewards).unsqueeze(0),
        torch.tensor([value * masks[-1]]).unsqueeze(0),
        gamma_matrix,
    )

    assert d_gamma.cpu().numpy() == approx(d_simple, 1e-5)
