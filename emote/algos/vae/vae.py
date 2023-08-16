import torch
import torch.nn.functional as F

from torch import nn, optim

from emote.callbacks import LossCallback
from emote.nn.initialization import normal_init_


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device,
        beta: float = 0.01,
    ):
        super().__init__()
        assert encoder.output_size == decoder.input_size
        self.latent_size = encoder.output_size
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.encoder.apply(normal_init_)
        self.decoder.apply(normal_init_)

    def forward(self, x, condition=None):
        mu, log_std = self.encoder(x, condition)
        var = torch.exp(log_std)
        eps = torch.randn_like(var).to(self.device)
        latent = eps.mul(var).add(mu)
        x_hat = self.decoder(latent, condition)
        x_hat = x_hat.view(x.size())
        return x_hat, mu, log_std, latent

    def loss(self, x, x_hat, mu, log_std):
        restore_loss = F.mse_loss(x_hat, x)
        var = torch.exp(log_std)
        kld = torch.sum(-log_std + (mu**2) * 0.5 + var, 1) - self.latent_size
        kl_loss = kld.mean()
        info = {"restore_loss": restore_loss, "kl_loss": kl_loss}
        loss = restore_loss + self.beta * kl_loss
        return loss, info


class VAELoss(LossCallback):
    def __init__(
        self,
        *,
        vae: VariationalAutoencoder,
        opt: optim.Optimizer,
        lr_schedule=None,
        max_grad_norm: float = 10.0,
        name: str = "vae",
        data_group: str = "default",
        input_key: str = "obs",
        conditioning_func=lambda _: None,
    ):
        super().__init__(
            name=name,
            optimizer=opt,
            lr_schedule=lr_schedule,
            network=vae,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
        )
        self.vae = vae
        self.conditioning_func = conditioning_func
        self._input_key = input_key

    def loss(self, observation, actions):

        condition = self.conditioning_func(observation[self._input_key])
        samples, dist_mean, dist_log_std, _ = self.vae.forward(actions, condition)
        loss, info = self.vae.loss(actions, samples, dist_mean, dist_log_std)
        self.log_scalar("genrl/restore_loss", torch.mean(info["restore_loss"]))
        self.log_scalar("genrl/kl_loss", torch.mean(info["kl_loss"]))
        return loss
