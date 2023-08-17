"""

This is an example code to train VAE models to generate action data. In order to train a VAE model, we need to obtain a
replay buffer of expert data. The replay buffer can come from previous RL training sessions. As an example, train a SAC
policy and store the replay buffer:

    python experiments/gym/train_lunar_lander.py --log-dir logs/ --bp-steps 50000 --export-memory

The '--export-memory' flag is needed to save the replay buffer after the training is finished.
Once the expert replay buffer is created, you can train the VAE model using the example below:

    python experiments/gym/train_vae.py --beta 0.004 --action-size 2 --observation-size 24
        --condition-size 24 --latent-size 1 --num-hidden-layer 4 --bp-steps 10000 --checkpoint-interval 9999
        --buffer-dir logs/ --buffer-filename default_export

The '--latent-size 1' learns a latent space to represent actions with only one dimension. Note that the original action
space is 2-dimensional.

"""


import argparse
import os

import numpy as np
import torch

from experiments.gym.train_lunar_lander import (
    create_complementary_callbacks,
    create_memory,
)
from torch.optim import Adam

from emote import Trainer
from emote.algos.vae.vae import VAELoss, VariationalAutoencoder
from emote.nn.layers import FullyConnectedDecoder, FullyConnectedEncoder
from emote.utils.spaces import BoxSpace, DictSpace, MDPSpace


def get_conditioning_fn(len_cond: int = 0):
    assert len_cond >= 0

    def conditioning_fn(a):
        return a[:, :len_cond]

    return conditioning_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="vae_training")

    parser.add_argument("--action-size", type=int)
    parser.add_argument("--observation-size", type=int)
    parser.add_argument("--observation-key", type=str, default="obs")
    parser.add_argument("--condition-size", type=int)
    parser.add_argument("--latent-size", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.01, help="VAE beta value")

    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--hidden-layer-size", type=int, default=256)
    parser.add_argument("--num-hidden-layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-3, help="The learning rate")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bp-steps", type=int, default=10000)

    parser.add_argument("--log-dir", type=str, default="logs/")

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100000)

    parser.add_argument("--buffer-dir", type=str, default="replay_buffers/")
    parser.add_argument("--buffer-filename", type=str, default="rl_buffer_export")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-run", type=str, default=None, help="Display name of the run for wandb."
    )

    arg = parser.parse_args()

    training_device = torch.device(arg.device)

    spaces = MDPSpace(
        rewards=None,
        actions=BoxSpace(dtype=np.float32, shape=(arg.action_size,)),
        state=DictSpace(
            spaces={
                arg.observation_key: BoxSpace(
                    dtype=np.float32, shape=(arg.observation_size,)
                )
            }
        ),
    )
    condition_fn = get_conditioning_fn(arg.condition_size)

    """Create the memory and pre-load it with some expert policy trajectories"""
    _, dataloader = create_memory(
        space=spaces,
        memory_size=4_000_000,
        len_rollout=1,
        batch_size=arg.batch_size,
        data_group="offline_data",
        preload_buffer=True,
        buffer_filename=os.path.join(arg.buffer_dir, arg.buffer_filename),
        device=training_device,
    )

    """Create the vae model """
    encoder = FullyConnectedEncoder(
        input_size=arg.action_size,
        output_size=arg.latent_size,
        condition_size=arg.condition_size,
        device=training_device,
        hidden_sizes=[arg.hidden_layer_size] * arg.num_hidden_layer,
    )
    decoder = FullyConnectedDecoder(
        latent_size=arg.latent_size,
        output_size=arg.action_size,
        condition_size=arg.condition_size,
        device=training_device,
        hidden_sizes=[arg.hidden_layer_size] * arg.num_hidden_layer,
    )

    vae = VariationalAutoencoder(
        encoder=encoder,
        decoder=decoder,
        device=training_device,
        beta=arg.beta,
    )

    cbs = [
        VAELoss(
            vae=vae,
            opt=Adam(vae.parameters(), lr=arg.lr),
            data_group="offline_data",
            conditioning_func=condition_fn,
            input_key=arg.observation_key,
        )
    ]

    """Creating the supplementary callbacks and adding them to the training callbacks """
    all_callbacks = create_complementary_callbacks(
        args=arg,
        logged_cbs=cbs,
        cbs_name_to_checkpoint=["vae"],
    )

    """Training """
    trainer = Trainer(all_callbacks, dataloader)
    trainer.train()
