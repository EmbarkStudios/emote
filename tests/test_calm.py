import torch

from torch import Tensor, nn

from emote.algorithms.calm import CALMReward, DiscriminatorLoss


class Discriminator(nn.Module):
    def __init__(self, input_size: int, latent_size: int, hidden_dims: list[int]):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip([input_size + latent_size] + hidden_dims, hidden_dims)
            ],
        )
        final_layers: list[nn.Module] = [nn.Linear(hidden_dims[-1], 1)]
        self.final_layer = nn.Sequential(*final_layers)

    def forward(self, x: Tensor, z: Tensor):
        data_in = torch.cat((x, z), dim=1)
        return self.final_layer(self.encoder(data_in))


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_dims: list[int], output_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.ReLU())
                for n_in, n_out in zip([input_size] + hidden_dims, hidden_dims)
            ],
        )
        final_layers: list[nn.Module] = [nn.Linear(hidden_dims[-1], output_size)]
        self.final_layer = nn.Sequential(*final_layers)

    def forward(self, x: Tensor):
        return self.final_layer(self.encoder(x))


def state_map_fn(obs: Tensor):
    return obs


def test_discriminator_loss():
    bs = 50
    length = 10
    data_dim = 35
    latent_dim = 3
    discriminator = Discriminator(
        input_size=data_dim * 2, latent_size=latent_dim, hidden_dims=[128, 128]
    )
    encoder = Encoder(
        input_size=(length + 1) * data_dim, hidden_dims=[128, 128], output_size=latent_dim
    )
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    loss = DiscriminatorLoss(
        discriminator=discriminator,
        encoder=encoder,
        animation_rollout_length=length,
        imitation_state_map_fn=state_map_fn,
        policy_state_map_fn=state_map_fn,
        grad_loss_weight=1,
        optimizer=discriminator_opt,
        lr_schedule=None,
        input_key="features",
        max_grad_norm=10.0,
    )
    animation_batch = {
        "batch_size": bs,
        "observation": {"features": torch.rand(bs * length, data_dim)},
        "next_observation": {"features": torch.rand(bs, data_dim)},
    }
    game_batch = {
        "batch_size": bs,
        "observation": {"features": torch.rand(bs, data_dim), "latent": torch.rand(bs, latent_dim)},
        "next_observation": {"features": torch.rand(bs, data_dim)},
    }
    assert loss.loss(animation_batch, game_batch) >= 0


def test_calm_reward():
    bs = 50
    rollout_length = 5
    data_dim = 35
    latent_dim = 3
    discriminator = Discriminator(
        input_size=data_dim * 2, latent_size=latent_dim, hidden_dims=[128, 128]
    )
    calm_reward = CALMReward(
        discriminator=discriminator,
        state_map_fn=state_map_fn,
        style_reward_weight=1.0,
        rollout_length=rollout_length,
        observation_key="features",
        data_group=None,
    )
    observation = {
        "features": torch.rand(bs * rollout_length, data_dim),
        "latent": torch.rand(bs * rollout_length, latent_dim),
    }
    next_observation = {"features": torch.rand(bs, data_dim)}
    reward = torch.rand(bs * rollout_length, 1)
    calm_reward.begin_batch(observation, next_observation, reward)
