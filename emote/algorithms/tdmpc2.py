import copy

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, optim

from emote import Trainer
from emote.algorithms.sac import AlphaLoss, GenericAgentProxy, PolicyLoss, soft_update_from_to
from emote.callback import Callback
from emote.callbacks import LossCallback
from emote.extra.schedules import Schedule
from emote.mixins.logging import LoggingMixin
from emote.typing import AgentId, DictObservation, DictResponse, EpisodeState
from emote.utils.gamma_matrix import split_rollouts
from emote.utils.tdmpc2_utils import soft_ce, two_hot_inv


class PolicyLossTDMPC2(PolicyLoss):
    def __init__(
        self,
        *,
        pi: nn.Module,
        ln_alpha: torch.tensor,
        q: nn.Module,
        opt: optim.Optimizer,
        rho: float,
        lr_schedule: Optional[optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 10.0,
        name: str = "policy",
        data_group: str = "default",
        log_per_param_weights=False,
        log_per_param_grads=False,
    ):
        super().__init__(
            pi=pi,
            ln_alpha=ln_alpha,
            q=q,
            opt=opt,
            lr_schedule=lr_schedule,
            max_grad_norm=max_grad_norm,
            name=name,
            data_group=data_group,
            log_per_param_weights=log_per_param_weights,
            log_per_param_grads=log_per_param_grads,
        )
        self.rho = rho

    def loss(self, zs):
        p_sample, logp_pi = self.policy(zs)

        q_pi = self.q1(p_sample, zs)
        alpha = torch.exp(self._ln_alpha).detach()

        # This is different compared to the normal loss fn
        # weigh temporally further time steps less
        rho = torch.pow(self.rho, torch.arange(len(q_pi)))
        policy_loss = ((alpha * logp_pi - q_pi).mean(dim=(1, 2)) * rho).mean()

        self.log_scalar("policy/q_pi", torch.mean(q_pi))
        self.log_scalar("policy/logp_pi", torch.mean(logp_pi))
        self.log_scalar("policy/alpha", torch.mean(alpha))
        assert policy_loss.dim() == 0
        return policy_loss


class QTargetTDMPC2(LoggingMixin, Callback):
    def __init__(
        self,
        *,
        pi: nn.Module,
        encoder: nn.Module,
        ln_alpha: torch.tensor,
        q: nn.Module,
        qt: Optional[nn.Module] = None,
        gamma: float = 0.99,
        reward_scale: float = 1.0,
        target_q_tau: float = 0.005,
        data_group: str = "default",
        roll_length: int = 1,
        use_terminal_masking: bool = False,
    ):
        super().__init__()

        self._order = 1  # this is to ensure that the data_group is prepared beforehand
        self.policy = pi
        self.encoder = encoder
        self.ln_alpha = ln_alpha
        self.q = q
        self.qt = copy.deepcopy(q) if qt is None else qt
        self.gamma = torch.tensor(gamma)
        self.reward_scale = reward_scale
        self.tau = target_q_tau
        self.data_group = data_group
        self.rollout_len = roll_length
        self.use_terminal_masking = use_terminal_masking

    def begin_batch(self, observation, next_observation, rewards):
        next_observations = []

        for key in observation.keys():
            # time steps / rollout len x num rollouts x obs dim
            observation[key] = split_rollouts(observation[key], self.rollout_len).transpose(0, 1)
            next = split_rollouts(next_observation[key], 1).transpose(0, 1)
            next_observations.append(torch.cat([observation[key][1:], next]))

        with torch.no_grad():
            next_z = self.encoder(*next_observations)
            next_p_sample, next_logp_pi = self.policy(next_z)
            next_value = self.qt(next_p_sample, next_z, return_type="min")

        scaled_reward = (
            split_rollouts(rewards, self.rollout_len).transpose(0, 1) * self.reward_scale
        )

        targets = scaled_reward + self.gamma * next_value

        self.log_scalar("training/next_logp_pi", torch.mean(next_logp_pi))
        self.log_scalar("training/min_next_q_target", torch.mean(next_value))
        self.log_scalar("training/scaled_reward", torch.mean(scaled_reward))
        self.log_scalar("training/q_target", torch.mean(targets))

        return {
            self.data_group: {"targets": targets, "next_z": next_z, "scaled_reward": scaled_reward}
        }

    def end_batch(self):
        super().end_batch()

        for q, qt in zip(self.q.qs, self.qt.qs):
            soft_update_from_to(q, qt, self.tau)


class AlphaLossTDMPC2(AlphaLoss):
    def __init__(
        self,
        *,
        pi: nn.Module,
        ln_alpha: torch.tensor,
        opt: optim.Optimizer,
        lr_schedule: optim.lr_scheduler._LRScheduler | None = None,
        n_actions: int,
        max_grad_norm: float = 10.0,
        max_alpha: float = 0.2,
        name: str = "alpha",
        data_group: str = "default",
        t_entropy: float | Schedule | None = None,
    ):
        super().__init__(
            pi=pi,
            ln_alpha=ln_alpha,
            opt=opt,
            lr_schedule=lr_schedule,
            n_actions=n_actions,
            max_grad_norm=max_grad_norm,
            name=name,
            data_group=data_group,
            t_entropy=t_entropy,
            max_alpha=max_alpha,
        )

    def loss(self, zs):
        with torch.no_grad():
            _, logp_pi = self.policy(zs)
            entropy = -logp_pi
            error = entropy - self.t_entropy.value
        alpha_loss = torch.mean(self.ln_alpha * error.detach())
        assert alpha_loss.dim() == 0
        self.log_scalar("loss/alpha_loss", alpha_loss)
        self.log_scalar("training/entropy", torch.mean(entropy).item())
        return alpha_loss


class TDMPC2Loss(LossCallback):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        dynamics: nn.Module,
        reward: nn.Module,
        policy: nn.Module,
        qs: nn.Module,
        opt: optim.Optimizer,
        lr_schedule: Optional[optim.lr_scheduler._LRScheduler] = None,
        rho: float,
        consistency_coef,
        reward_coef,
        value_coef,
        max_grad_norm: float = 10.0,
        name: str = "tdmpc2",
        data_group: str = "default",
        log_per_param_weights=False,
        log_per_param_grads=False,
        rollout_length=1,
        horizon=3,
        device="cpu",
        bin_min=-10,
        bin_max=10,
        num_bins=101,
    ):
        super().__init__(
            name=name,
            network=None,
            optimizer=opt,
            lr_schedule=lr_schedule,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
            log_per_param_weights=log_per_param_weights,
            log_per_param_grads=log_per_param_grads,
        )

        self._order = -1  # to make sure the updates executes before the policy
        self.encoder = encoder
        self.dynamics = dynamics
        self.reward = reward
        self.policy = policy
        self.qs = qs

        self.rho = rho
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.rollout_length = rollout_length
        self.horizon = horizon
        self.device = device
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.num_bins = num_bins

        self._named_parameters = {}
        mappings = {
            "encoder": self.encoder,
            "dynamics": self.dynamics,
            "reward": self.reward,
            "qs": self.qs,
        }
        for name, network in mappings.items():
            network_named_params = {
                f"{name}.{n}": p
                for n, p in network.named_parameters(recurse=True)
                if any(p is p_ for p_ in network.parameters())
            }
            self._named_parameters.update(network_named_params)

    def loss(self, observation, actions, scaled_reward, rewards, next_z, targets):
        actions = split_rollouts(actions, self.rollout_length).transpose(0, 1)
        rewards = split_rollouts(rewards, self.rollout_length).transpose(0, 1)

        # encode first time step observations
        z = self.encoder(*[observation[key][0] for key in observation.keys()])

        # latent rollout
        # z(0) = h(s0)
        # z(t+1) = dynamics(zt, at)
        zs = torch.empty(
            self.horizon + 1,
            z.shape[0],
            z.shape[1],
            device=self.device,
        )
        zs[0] = z

        consistency_loss = 0
        # difference between next state from dynamics model vs actual
        for t in range(self.horizon):
            z = self.dynamics(torch.cat([z, actions[t]], dim=-1))
            zs[t + 1] = z

            consistency_loss += F.mse_loss(z, next_z[t]) * self.rho**t

        _zs = zs[:-1]
        qs = self.qs(_zs, actions, return_type="all")
        reward_preds = self.reward(torch.cat([_zs, actions], dim=-1))

        reward_loss, value_loss = 0, 0
        for t in range(self.horizon):
            reward_loss += (
                soft_ce(
                    reward_preds[t], scaled_reward[t], self.bin_min, self.bin_max, self.num_bins
                ).mean()
                * self.rho**t
            )

            for q in range(self.qs.num_qs):
                value_loss += (
                    soft_ce(qs[q][t], targets[t], self.bin_min, self.bin_max, self.num_bins).mean()
                    * self.rho**t
                )

        consistency_loss *= 1 / self.horizon
        reward_loss *= 1 / self.horizon
        value_loss *= 1 / (self.horizon * self.qs.num_qs)

        self.log_scalar(f"loss/{self.name}_consistency_loss", consistency_loss)
        self.log_scalar(f"loss/{self.name}_reward_loss", reward_loss)
        self.log_scalar(f"loss/{self.name}_value_loss", value_loss)

        loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.value_coef * value_loss
        )

        return loss, zs

    def backward(self, *args, **kwargs):
        self.optimizer.zero_grad()
        loss, zs = self.loss(*args, **kwargs)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.parameters, self._max_grad_norm)
        self.optimizer.step()
        self.lr_schedule.step()
        self.log_scalar(f"loss/{self.name}_lr", self.lr_schedule.get_last_lr()[0])
        self.log_scalar(f"loss/{self.name}_total_loss", loss.item())
        self.log_scalar(f"loss/{self.name}_gradient_norm", grad_norm.item())

        if self._log_per_param_weights or self._log_per_param_grads:
            self.log_per_param_weights_and_grads()

        return {self.data_group: {"zs": zs.detach()}}


class TDMPC2Proxy(GenericAgentProxy):
    def __init__(
        self,
        input_keys: tuple,
        output_keys: tuple,
        encoder,
        dynamics_model,
        reward_model,
        policy,
        qs,
        num_p_traj,
        horizon,
        action_dim,
        discount,
        max_std=2,
        min_std=0.05,
        num_traj_total=512,
        num_iter=6,
        num_k=64,
        temperature=0.5,
        bin_min=-10,
        bin_max=10,
        num_bins=101,
        device="cpu",
    ):
        super().__init__(
            policy=policy,
            device=device,
            input_keys=input_keys,
            output_keys=output_keys,
        )
        self.encoder = encoder
        self.dynamics = dynamics_model
        self.reward = reward_model
        self.qs = qs
        self.num_p_traj = num_p_traj
        self.horizon = horizon
        self.discount = discount
        self.device = device
        self.max_std = max_std
        self.min_std = min_std
        self.num_traj_total = num_traj_total
        self.num_iter = num_iter
        self.num_k = num_k
        self.temperature = temperature
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.num_bins = num_bins

        self.action_dim = action_dim

        self._prev_mean = None

    def __call__(self, observations: dict[AgentId, DictObservation]) -> dict[AgentId, DictResponse]:
        """Returns the actions."""
        # The network takes observations of size batch x obs for each observation space.
        assert len(observations) > 0, "Observations must not be empty."

        active_agents = [
            agent_id
            for agent_id, obs in observations.items()
            if obs.episode_state not in self._end_states
        ]

        episode_states = np.array(
            [observations[agent_id].episode_state for agent_id in active_agents]
        )

        tensor_obs_list = [None] * len(self.input_keys)
        for input_key in self.input_keys:
            np_obs = np.array(
                [observations[agent_id].array_data[input_key] for agent_id in active_agents]
            )

            if self._spaces is not None:
                shape = (np_obs.shape[0],) + self._spaces.state.spaces[input_key].shape
                if shape != np_obs.shape:
                    np_obs = np.reshape(np_obs, shape)

            tensor_obs = torch.tensor(np_obs).to(self.device)
            index = self.input_keys.index(input_key)
            tensor_obs_list[index] = tensor_obs

        outputs = self._plan(*tensor_obs_list, episode_states)
        outputs = {key: outputs[i].detach().cpu().numpy() for i, key in enumerate(self.output_keys)}

        agent_data = [
            (agent_id, DictResponse(list_data={}, scalar_data={})) for agent_id in active_agents
        ]

        for i, (_, response) in enumerate(agent_data):
            for k, data in outputs.items():
                response.list_data[k] = data[i]

        return dict(agent_data)

    def _make_copies(self):
        self.encoder_copy = copy.deepcopy(self.encoder)
        self.dynamics_copy = copy.deepcopy(self.dynamics)
        self.reward_copy = copy.deepcopy(self.reward)
        self.policy_copy = copy.deepcopy(self.policy)
        self.qs_copy = copy.deepcopy(self.qs)

        self.encoder_copy.train(False)
        self.dynamics_copy.train(False)
        self.reward_copy.train(False)
        self.policy_copy.train(False)
        self.qs_copy.train(False)

    @torch.no_grad()
    def _estimate_value(self, z, actions):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        reward, discount = 0, 1

        for t in range(self.horizon):
            r = self.reward_copy(torch.cat([z, actions[t]], dim=-1))
            r = two_hot_inv(r, self.bin_min, self.bin_max, self.num_bins)
            z = self.dynamics_copy(torch.cat([z, actions[t]], dim=-1))

            reward += discount * r
            discount *= self.discount

        actions = self.policy_copy(z, torch.randn(self.action_dim))
        qs = self.qs_copy(z, actions, return_type="avg")

        return reward + discount * qs

    @torch.no_grad()
    def _plan(self, obs, episode_states):
        self._make_copies()

        # encode observation
        z = self.encoder_copy(obs)

        batch_dim = z.shape[0]
        z = z.unsqueeze(1)

        # sample num_p_traj of length horizon using the policy and the dynamics model
        if self.num_p_traj > 0:
            zs = z.expand(-1, self.num_p_traj, -1)

            # horizon time steps x num agents x num traj x actions
            p_actions = torch.empty(
                self.horizon,
                batch_dim,
                self.num_p_traj,
                self.action_dim,
                device=self.device,
            )

            # imagine rollouts: starting from the actual observation, generate actions with the policy and next state with the dynamics model
            for t in range(self.horizon):
                eps = torch.randn(self.action_dim)
                a_t = self.policy_copy(zs, eps)
                p_actions[t] = a_t

                zs = self.dynamics_copy(torch.cat([zs, a_t], dim=-1))

        # Initialize state and parameters
        z = z.expand(-1, self.num_traj_total, -1)
        mean = torch.zeros(self.horizon, batch_dim, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(
            self.horizon, batch_dim, self.action_dim, device=self.device
        )

        if self._prev_mean is not None:
            for i, state in enumerate(episode_states):
                if state != EpisodeState.INITIAL:
                    mean[:-1, i] = self._prev_mean[1:, i]

        actions = torch.empty(
            self.horizon,
            batch_dim,
            self.num_traj_total,
            self.action_dim,
            device=self.device,
        )

        if self.num_p_traj > 0:
            actions[:, :, : self.num_p_traj] = p_actions

        # MPPI iteration
        for i in range(self.num_iter):
            # sample num_traj_total - num_p_traj trajectories using mean and std
            actions[:, :, self.num_p_traj :] = torch.tanh(
                mean.unsqueeze(2)
                + std.unsqueeze(2)
                * torch.randn(
                    self.horizon,
                    batch_dim,
                    self.num_traj_total - self.num_p_traj,
                    self.action_dim,
                    device=self.device,
                )
            )

            # estimate values
            values = self._estimate_value(z, actions)

            # get best actions
            top_k_ind = torch.topk(values.squeeze(-1), self.num_k, dim=1).indices
            top_k_values = values.gather(1, top_k_ind.unsqueeze(-1))
            max_value = torch.max(top_k_values)
            # self.log_scalar("tdmpc2/max_value", torch.mean(max_value).item())

            top_k_actions = actions[:, torch.arange(batch_dim)[:, None], top_k_ind, :]

            # calculate scores
            # num agents x num k x 1
            score = torch.exp(self.temperature * (top_k_values - max_value))
            score /= torch.sum(score, 1).unsqueeze(-1)

            # update mean and std
            mean = torch.sum(score * top_k_actions, dim=2) / (score.sum(dim=1) + 1e-9)
            std = torch.sqrt(
                torch.sum(score * (top_k_actions - mean.unsqueeze(2)) ** 2, dim=2)
                / (score.sum(dim=1) + 1e-9)
            ).clamp(self.min_std, self.max_std)

        # select actions
        score = score.squeeze(-1).cpu().numpy()
        selected = np.empty(batch_dim, dtype=int)
        for i in range(batch_dim):
            selected[i] = np.random.choice(np.arange(self.num_k), p=score[i])
        actions = top_k_actions[0, torch.arange(batch_dim), selected, :]
        self._prev_mean = mean

        return (torch.tanh(actions),)
