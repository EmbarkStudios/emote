import torch

from torch import Tensor, nn

from emote.algorithms.amp import AMPReward, DiscriminatorLoss, gradient_loss_function


class Discriminator(nn.Module):
    def __init__(self, input_size: int, hidden_dims: list[int]):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip([input_size] + hidden_dims, hidden_dims)
            ],
        )
        final_layers: list[nn.Module] = [nn.Linear(hidden_dims[-1], 1)]
        self.final_layer = nn.Sequential(*final_layers)

    def forward(self, x: Tensor):
        return self.final_layer(self.encoder(x))


def state_map_fn(obs: Tensor):
    return obs


def test_gradient_loss():

    x = torch.ones(10, 3, requires_grad=True)
    x = x * torch.rand(10, 3)
    y = torch.sum(4 * x * x + torch.sin(x), dim=1)

    grad1 = gradient_loss_function(y, x)
    y_dot = 8 * x + torch.cos(x)
    grad2 = torch.mean(torch.sum(y_dot * y_dot, dim=1))

    assert abs(grad1.item() - grad2.item()) < 0.001


def test_discriminator_loss():
    discriminator = Discriminator(input_size=20, hidden_dims=[128, 128])
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    loss = DiscriminatorLoss(
        discriminator=discriminator,
        state_map_fn=state_map_fn,
        grad_loss_weight=1,
        optimizer=discriminator_opt,
        lr_schedule=None,
        input_key="features",
        max_grad_norm=10.0,
    )
    batch1 = {
        "batch_size": 30,
        "observation": {"features": torch.rand(30, 10)},
        "next_observation": {"features": torch.rand(30, 10)},
    }
    batch2 = {
        "batch_size": 30,
        "observation": {"features": torch.rand(30, 10)},
        "next_observation": {"features": torch.rand(30, 10)},
    }
    assert loss.loss(batch1, batch2) >= 0


def test_amp_reward():
    discriminator = Discriminator(input_size=20, hidden_dims=[128, 128])
    amp_reward = AMPReward(
        discriminator=discriminator,
        state_map_fn=state_map_fn,
        style_reward_weight=1.0,
        rollout_length=1,
        observation_key="features",
        data_group=None,
    )
    observation = {"features": torch.rand(30, 10)}
    next_observation = {"features": torch.rand(30, 10)}
    reward = torch.rand(30, 1)
    amp_reward.begin_batch(observation, next_observation, reward)
