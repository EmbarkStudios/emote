import argparse

import numpy as np
import torch

from experiments.gym.train_lunar_lander import (
    _make_env,
    create_complementary_callbacks,
    create_train_callbacks,
)
from experiments.gym.train_vae import get_conditioning_fn
from gymnasium.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper

from emote import Trainer
from emote.algos.genrl.nn import GenRLPolicy, GenRLQNet
from emote.algos.genrl.proxies import MemoryProxyWithEncoder
from emote.algos.vae.nn import DecoderWrapper, EncoderWrapper
from emote.memory import MemoryLoader
from emote.memory.builder import DictObsNStepTable
from emote.sac import FeatureAgentProxy
from emote.utils.spaces import BoxSpace, MDPSpace


def create_memory(
    encoder: EncoderWrapper,
    space: MDPSpace,
    memory_size: int,
    len_rollout: int,
    batch_size: int,
    data_group: str,
    device: torch.device,
    observation_key: str = "obs",
):

    use_terminal_masking = True

    table = DictObsNStepTable(
        spaces=space,
        use_terminal_column=use_terminal_masking,
        maxlen=memory_size,
        device=device,
    )

    memory_proxy = MemoryProxyWithEncoder(
        table=table,
        encoder=encoder,
        minimum_length_threshold=len_rollout,
        use_terminal=use_terminal_masking,
        input_key=observation_key,
    )

    data_loader = MemoryLoader(
        table=table,
        rollout_count=batch_size // len_rollout,
        rollout_length=len_rollout,
        size_key="batch_size",
        data_group=data_group,
    )

    return memory_proxy, data_loader


def create_actor_critic_agents(
    args,
    num_obs: int,
    num_actions: int,
    num_latent: int,
    decoder: DecoderWrapper,
    encoder: EncoderWrapper,
    init_alpha: float = 0.01,
):

    device = args.device

    hidden_dims = [args.hidden_layer_size] * arg.num_hidden_layer
    q1 = GenRLQNet(encoder, num_obs, num_actions, hidden_dims).to(device)
    q2 = GenRLQNet(encoder, num_obs, num_actions, hidden_dims).to(device)
    policy = GenRLPolicy(decoder, num_obs, num_latent, hidden_dims).to(device)

    policy_proxy = FeatureAgentProxy(policy, device=device)

    logn_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=device)

    return q1, q2, policy_proxy, logn_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="genrl")
    parser.add_argument("--observation-key", type=str, default="obs")
    parser.add_argument("--log-dir", type=str, default="logs/")
    parser.add_argument(
        "--vae-checkpoint-dir", type=str, default="checkpoints/training"
    )
    parser.add_argument("--vae-checkpoint-index", type=int, default=0)
    parser.add_argument("--vae-latent-size", type=int, default=3)
    parser.add_argument("--condition-size", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--hidden-layer-size", type=int, default=256)
    parser.add_argument("--num-hidden-layer", type=int, default=2)
    parser.add_argument("--actor-lr", type=float, default=8e-3, help="Policy lr")
    parser.add_argument("--critic-lr", type=float, default=8e-3, help="Q-function lr")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bp-steps", type=int, default=100000)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-run", type=str, default=None, help="wandb run name")

    arg = parser.parse_args()

    training_device = torch.device(arg.device)

    """Creating a vector of Gym environments """
    gym_wrapper = DictGymWrapper(
        AsyncVectorEnv([_make_env() for _ in range(arg.num_envs)])
    )

    number_of_actions = gym_wrapper.dict_space.actions.shape[0]
    number_of_obs = list(gym_wrapper.dict_space.state.spaces.values())[0].shape[0]

    condition_func = get_conditioning_fn(arg.condition_size)

    """Create the decoder wrapper"""
    action_latent_size = arg.vae_latent_size

    vae_decoder = DecoderWrapper(
        latent_size=action_latent_size,
        output_size=number_of_actions,
        condition_size=arg.condition_size,
        condition_fn=condition_func,
        hidden_sizes=[arg.hidden_layer_size] * arg.num_hidden_layer,
        device=training_device,
    )

    """Create the encoder wrapper"""
    vae_encoder = EncoderWrapper(
        latent_size=action_latent_size,
        action_size=number_of_actions,
        hidden_sizes=[arg.hidden_layer_size] * arg.num_hidden_layer,
        condition_size=arg.condition_size,
        condition_fn=condition_func,
        device=training_device,
    )

    checkpoint_filename = f"{arg.vae_checkpoint_dir}.{arg.vae_checkpoint_index}.tar"
    state_dict = torch.load(checkpoint_filename, map_location=training_device)

    state = state_dict["callback_state_dicts"]["vae"]

    vae_encoder.load_state_dict(state.pop("network_state_dict"))
    for param in vae_encoder.parameters():
        param.requires_grad = False

    state_dict = torch.load(checkpoint_filename)
    state = state_dict["callback_state_dicts"]["vae"]
    vae_decoder.load_state_dict(state.pop("network_state_dict"))
    for param in vae_decoder.parameters():
        param.requires_grad = False

    """Create agent and the Q-functions"""
    qnet1, qnet2, agent_proxy, ln_alpha = create_actor_critic_agents(
        decoder=vae_decoder,
        encoder=vae_encoder,
        args=arg,
        num_actions=number_of_actions,
        num_latent=action_latent_size,
        num_obs=number_of_obs,
    )

    spaces = MDPSpace(
        rewards=gym_wrapper.dict_space.rewards,
        actions=BoxSpace(dtype=np.float32, shape=(action_latent_size,)),
        state=gym_wrapper.dict_space.state,
    )

    gym_memory, dataloader = create_memory(
        space=spaces,
        encoder=vae_encoder,
        memory_size=4_000_000,
        len_rollout=arg.rollout_length,
        batch_size=arg.batch_size,
        data_group="rl_buffer",
        device=training_device,
    )

    train_callbacks = create_train_callbacks(
        args=arg,
        q1=qnet1,
        q2=qnet2,
        policy_proxy=agent_proxy,
        ln_alpha=ln_alpha,
        env=gym_wrapper,
        memory_proxy=gym_memory,
        data_group="rl_buffer",
    )

    """Creating the complementary callbacks and starting the training"""
    callbacks = create_complementary_callbacks(
        args=arg,
        logged_cbs=train_callbacks,
    )

    trainer = Trainer(callbacks, dataloader)
    trainer.train()
