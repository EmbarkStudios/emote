import argparse

from functools import partial

import numpy as np
import torch

from experiments.gym.train_lunar_lander import (
    _make_env,
    create_complementary_callbacks,
    create_memory,
)
from gymnasium.vector import AsyncVectorEnv
from tests.gym import DictGymWrapper
from tests.gym.collector import ThreadedGymCollector
from torch import Tensor, nn
from torch.optim import Adam

from emote import Trainer
from emote.algorithms.tdmpc2 import (
    AlphaLossTDMPC2,
    PolicyLossTDMPC2,
    QTargetTDMPC2,
    TDMPC2Loss,
    TDMPC2Proxy,
)
from emote.nn import GaussianPolicyHead
from emote.nn.initialization import constant_init_, trunc_normal_init_, xavier_uniform_init_
from emote.utils.tdmpc2_utils import SimNorm, two_hot_inv


class GaussianMlpPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        input_key="features",
        activation=nn.Mish(),
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.LayerNorm(n_out), activation)
                for n_in, n_out in zip([observation_dim] + hidden_dims, hidden_dims)
            ],
        )

        self.policy = GaussianPolicyHead(hidden_dims[-1], action_dim)

        self.encoder.apply(trunc_normal_init_)
        self.policy.apply(partial(xavier_uniform_init_, gain=0.01))

        self.input_key = input_key
        self.action_dim = action_dim

    def forward(self, obs: Tensor, epsilon: Tensor = None) -> Tensor:
        output = self.policy(self.encoder(obs), epsilon)

        if isinstance(output, tuple) and len(output) > 1:
            sample, log_prob = output
            # sample, log_prob = self.policy(self.encoder(obs))
            # TODO: Investigate the log_prob() logic of the pytorch distribution code.
            # The change below shouldn't be needed but significantly improves training
            # stability when training lunar lander.
            log_prob = log_prob.clamp(min=-2)
            return sample, log_prob
        else:
            return output


class MLP(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dims, dropout=0.0, activation=nn.Mish(), final_init_fn=None
    ):
        super().__init__()

        dropout = [dropout] * len(hidden_dims) if not isinstance(dropout, list) else dropout

        activation = (
            [activation] * (len(hidden_dims) + 1)
            if not isinstance(activation, list)
            else activation
        )

        self.encoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(n_in, n_out),
                    nn.Dropout(d) if d else nn.Identity(),
                    nn.LayerNorm(n_out),
                    act,
                )
                for n_in, n_out, d, act in zip(
                    [in_dim] + hidden_dims, hidden_dims, dropout, activation
                )
            ]
        )

        if isinstance(activation[-1], nn.Mish):
            self.final_layer = nn.Linear(hidden_dims[-1], out_dim)
        else:
            self.final_layer = nn.Sequential(
                nn.Linear(hidden_dims[-1], out_dim), nn.LayerNorm(out_dim), activation[-1]
            )

        self.encoder.apply(trunc_normal_init_)
        self.final_layer.apply(final_init_fn if final_init_fn is not None else trunc_normal_init_)

    def forward(self, x):
        return self.final_layer(self.encoder(x))


class Q(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        out_dim,
        hidden_dims: list[int],
        dropout,
        final_init_fn,
        num_qs=1,
        bin_min=-10,
        bin_max=10,
    ):
        super().__init__()
        self.obs_d = observation_dim
        self.act_d = action_dim
        self.num_qs = num_qs
        self.qs = [
            MLP(
                observation_dim + action_dim,
                out_dim,
                hidden_dims,
                dropout,
                final_init_fn=final_init_fn,
            )
            for _ in range(self.num_qs)
        ]

        self.bin_min = bin_min
        self.bin_max = bin_max
        self.num_bins = out_dim

    def parameters(self):
        trainable_params = list()
        for q in self.qs:
            trainable_params += list(q.parameters())

        return trainable_params

    def train(self, mode=True):
        super().train(mode)
        for q in self.qs:
            q.train(mode)
        return self

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for i, q in enumerate(self.qs):
            for name, param in q.named_parameters(prefix, recurse):
                yield f"q{i+1}_{name}", param

    def forward(self, action: Tensor, obs: Tensor, return_type="avg") -> Tensor:
        assert return_type in ["avg", "min", "all"]

        x = torch.cat([obs, action], dim=-1)

        if return_type == "all":
            return [q(x) for q in self.qs]

        sampled_indices = np.random.choice(self.num_qs, 2, replace=False)

        q1 = self.qs[sampled_indices[0]](x)
        q2 = self.qs[sampled_indices[1]](x)

        q1 = two_hot_inv(q1, self.bin_min, self.bin_max, self.num_bins)
        q2 = two_hot_inv(q2, self.bin_min, self.bin_max, self.num_bins)
        return 0.5 * (q1 + q2) if return_type == "avg" else torch.min(q1, q2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="tdmpc2")
    parser.add_argument("--log-dir", type=str, default="/runs/tdmpc2")
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--actor-lr", type=float, default=8e-3, help="The policy learning rate")
    parser.add_argument("--critic-lr", type=float, default=8e-3, help="Q-function learning rate")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bp-steps", type=int, default=10000)
    parser.add_argument("--export-memory", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--wandb-run",
        type=str,
        default=None,
        help="Short display name of run for the W&B UI. Randomly generated by default.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--reward-scale", type=int, default=0.1)
    parser.add_argument("--gamma", type=int, default=0.99)

    # encoder
    parser.add_argument("--latent-size", type=int, default=256)  # 512
    parser.add_argument("--encoder-hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-hidden-layers", type=int, default=1)
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0)

    # MLPs
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)  # 512
    parser.add_argument("--mlp-hidden-layers", type=int, default=1)  # 2

    parser.add_argument("--num-qs", type=int, default=2)  # 5
    parser.add_argument("--q-dropout", type=float, default=0.01)

    # planning
    parser.add_argument("--horizon", type=int, default=5)  # 3
    parser.add_argument("--num-p-traj", type=int, default=24)
    parser.add_argument("--num-traj-total", type=int, default=128)  # 512
    parser.add_argument("--num-iter", type=int, default=6)
    parser.add_argument("--num-k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-std", type=float, default=20.0)
    parser.add_argument("--min-std", type=float, default=0.05)

    # loss
    parser.add_argument("--consistency-coef", type=float, default=10.0)
    parser.add_argument("--reward-coef", type=float, default=0.5)
    parser.add_argument("--value-coef", type=float, default=1.0)

    parser.add_argument("--rho", type=float, default=0.5)

    parser.add_argument("--two-hot-bins", type=int, default=101)
    parser.add_argument("--two-hot-bin-min", type=int, default=-10)
    parser.add_argument("--two-hot-bin-max", type=int, default=10)
    parser.add_argument("--simnorm-dim", type=int, default=8)

    input_args = parser.parse_args()
    training_device = torch.device(input_args.device)

    len_rollout = input_args.horizon
    batch_size = input_args.batch_size * input_args.horizon
    init_alpha = 0.01

    """Creating a vector of Gym environments """
    gym_wrapper = DictGymWrapper(AsyncVectorEnv([_make_env() for _ in range(input_args.num_envs)]))
    number_of_actions = gym_wrapper.dict_space.actions.shape[0]
    number_of_obs = list(gym_wrapper.dict_space.state.spaces.values())[0].shape[0]

    """Creating the memory and the dataloader"""
    gym_memory_proxy, dataloader = create_memory(
        space=gym_wrapper.dict_space,
        memory_size=4_000_000,
        len_rollout=len_rollout,
        batch_size=batch_size,
        data_group="default",
        device=training_device,
    )

    """Create a memory exporter if needed"""
    if input_args.export_memory:
        from emote.memory.memory import MemoryExporterProxyWrapper

        gym_memory_proxy = MemoryExporterProxyWrapper(
            memory=gym_memory_proxy,
            target_memory_name=dataloader.data_group,
            inf_steps_per_memory_export=10000000,
            experiment_root_path=input_args.log_dir,
            min_time_per_export=0,
        )

    """Creating the actor (policy) and critics (the Q-functions) agents """
    qs = Q(
        input_args.latent_size,
        number_of_actions,
        input_args.two_hot_bins,
        [input_args.mlp_hidden_dim] * input_args.mlp_hidden_layers,
        dropout=[input_args.q_dropout] + [0.0] * (input_args.mlp_hidden_layers - 1),
        final_init_fn=partial(constant_init_, val=0),
        num_qs=input_args.num_qs,
    )

    policy = GaussianMlpPolicy(
        input_args.latent_size,
        number_of_actions,
        [input_args.mlp_hidden_dim] * input_args.mlp_hidden_layers,
    )

    encoder_z = MLP(
        number_of_obs,
        input_args.latent_size,
        [input_args.encoder_hidden_dim] * input_args.encoder_hidden_layers,
        activation=[nn.Mish()] * input_args.encoder_hidden_layers
        + [SimNorm(input_args.simnorm_dim)],
    )

    dynamics = MLP(
        input_args.latent_size + number_of_actions,
        input_args.latent_size,
        [input_args.mlp_hidden_dim] * input_args.mlp_hidden_layers,
        activation=[nn.Mish()] * input_args.mlp_hidden_layers + [SimNorm(input_args.simnorm_dim)],
    )

    reward = MLP(
        input_args.latent_size + number_of_actions,
        input_args.two_hot_bins,
        [input_args.mlp_hidden_dim] * input_args.mlp_hidden_layers,
        final_init_fn=partial(constant_init_, val=0),
    )

    qs = qs.to(training_device)
    policy = policy.to(training_device)
    encoder_z = encoder_z.to(training_device)
    dynamics = dynamics.to(training_device)
    reward = reward.to(training_device)

    params = [
        {
            "params": encoder_z.parameters(),
            "lr": input_args.actor_lr * input_args.encoder_lr_scale,
        },
        {"params": dynamics.parameters()},
        {"params": reward.parameters()},
        {"params": qs.parameters()},
    ]

    tdmpc2_opt = Adam(params, input_args.actor_lr)

    proxy = TDMPC2Proxy(
        input_keys=("obs",),
        output_keys=("actions",),
        encoder=encoder_z,
        dynamics_model=dynamics,
        reward_model=reward,
        policy=policy,
        qs=qs,
        num_p_traj=input_args.num_p_traj,
        horizon=input_args.horizon,
        action_dim=number_of_actions,
        discount=input_args.gamma,
        max_std=input_args.max_std,
        min_std=input_args.min_std,
        num_traj_total=input_args.num_traj_total,
        num_iter=input_args.num_iter,
        num_k=input_args.num_k,
        temperature=input_args.temperature,
        bin_min=input_args.two_hot_bin_min,
        bin_max=input_args.two_hot_bin_max,
        num_bins=input_args.two_hot_bins,
        device=training_device,
    )

    ln_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=training_device)

    training_cbs = [
        PolicyLossTDMPC2(
            pi=policy,
            ln_alpha=ln_alpha,
            q=qs,
            opt=Adam(policy.parameters(), lr=input_args.actor_lr),
            max_grad_norm=input_args.max_grad_norm,
            data_group=dataloader.data_group,
            rho=input_args.rho,
            log_per_param_grads=True,
            log_per_param_weights=True,
        ),
        AlphaLossTDMPC2(
            pi=policy,
            ln_alpha=ln_alpha,
            opt=Adam([ln_alpha], lr=input_args.actor_lr),
            n_actions=number_of_actions,
            max_grad_norm=input_args.max_grad_norm,
            max_alpha=10.0,
            data_group=dataloader.data_group,
        ),
        QTargetTDMPC2(
            pi=policy,
            encoder=encoder_z,
            ln_alpha=ln_alpha,
            q=qs,
            reward_scale=input_args.reward_scale,
            data_group=dataloader.data_group,
            roll_length=len_rollout,
        ),
        TDMPC2Loss(
            encoder=encoder_z,
            dynamics=dynamics,
            reward=reward,
            policy=policy,
            qs=qs,
            opt=tdmpc2_opt,
            rho=input_args.rho,
            consistency_coef=input_args.consistency_coef,
            reward_coef=input_args.reward_coef,
            value_coef=input_args.value_coef,
            rollout_length=len_rollout,
            horizon=input_args.horizon,
            device=training_device,
            log_per_param_grads=True,
            log_per_param_weights=True,
            num_bins=input_args.two_hot_bins,
            bin_min=input_args.two_hot_bin_min,
            bin_max=input_args.two_hot_bin_max,
        ),
        ThreadedGymCollector(
            gym_wrapper,
            proxy,
            gym_memory_proxy,
            warmup_steps=batch_size,
            render=False,
        ),
    ]

    """Creating the supplementary callbacks and adding them to the training callbacks """
    all_callbacks = create_complementary_callbacks(args=input_args, logged_cbs=training_cbs)

    """Training """
    trainer = Trainer(all_callbacks, dataloader)
    trainer.train()
