import torch

from torch.optim import Adam

from emote.algos.vae.vae import VAELoss, VariationalAutoencoder
from emote.nn.layers import FullyConnectedDecoder, FullyConnectedEncoder


def get_conditioning_fn(len_cond: int = 0):
    assert len_cond >= 0

    def conditioning_fn(a):
        return a[:, :len_cond]

    return conditioning_fn


def test_vae_training():
    action_dim = 10
    obs_dim = 20
    latent_dim = 3
    batch_size = 100
    device = torch.device("cpu")
    hidden_layer_dim = [256] * 3

    actions = torch.rand(batch_size, action_dim)
    obs = torch.rand(batch_size, obs_dim)
    beta = 0.001

    encoder = FullyConnectedEncoder(
        input_size=action_dim,
        output_size=latent_dim,
        condition_size=obs_dim,
        device=device,
        hidden_sizes=hidden_layer_dim,
    )

    decoder = FullyConnectedDecoder(
        latent_size=latent_dim,
        output_size=action_dim,
        condition_size=obs_dim,
        device=device,
        hidden_sizes=hidden_layer_dim,
    )

    vae = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        device=device,
        beta=beta,
    )

    cfn = get_conditioning_fn(obs_dim)

    vae_loss = VAELoss(
        vae=vae,
        opt=Adam(vae.parameters(), lr=0.001),
        conditioning_func=cfn,
    )

    vae_loss.optimizer.zero_grad()

    loss = vae_loss.loss(default={"actions": actions, "observation": {"obs": obs}})
    loss_v1 = loss.item()

    loss.backward()
    vae_loss.optimizer.step()

    loss = vae_loss.loss(default={"actions": actions, "observation": {"obs": obs}})
    loss_v2 = loss.item()
    assert loss_v1 > loss_v2
