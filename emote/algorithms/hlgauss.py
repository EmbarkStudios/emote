from typing import Optional

import torch
import torch.nn.functional as F

from torch import Tensor, nn, optim

from emote.callbacks.loss import LossCallback


class LogitNet(nn.Module):
    """The QNet assumes that the input network has a num_bins property."""

    def __init__(self, num_bins):
        super().__init__()
        self.num_bins = num_bins


class QNet(nn.Module):
    """The HL Gauss QNet needs to output both the q-value based on the input
    and to convert logits to q."""

    def __init__(
        self,
        logit_net: LogitNet,
        min_value: float,
        max_value: float,
    ):
        super().__init__()
        self.logit_net = logit_net
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = logit_net.num_bins
        support = torch.linspace(min_value, max_value, logit_net.num_bins + 1, dtype=torch.float32)
        self.register_buffer("centers", (support[:-1] + support[1:]) / 2)

    def forward(self, *args, **kwargs) -> Tensor:
        logits = self.logit_net(*args, **kwargs)
        out = self.q_from_logit(logits)
        assert 1 == out.shape[1]
        return out

    def q_from_logit(self, logits: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=1)
        out = torch.sum(probs * self.centers, dim=-1)
        out.unsqueeze_(1)
        return out


class HLGaussLoss(nn.Module):
    r"""A HLGauss loss as described by Imani and White.

    Code from Google Deepmind's
    https://arxiv.org/pdf/2403.03950v1.pdf.

    :param min_value (float): Minimal value of the range of target bins.
    :param max_value (float): Maximal value of the range of target bins.
    :param num_bins (int): Number of bins.
    :param sigma (float): Standard deviation of the Gaussian used to
        convert regression targets to distributions.
    """

    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        support = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32)
        self.register_buffer(
            "support",
            support,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t_target = self.transform_to_probs(target)
        assert logits.shape == t_target.shape
        return F.cross_entropy(logits, t_target)

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        cdf_evals = torch.special.erf(
            (self.support - target.squeeze(1).unsqueeze(-1))
            / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)


class QLoss(LossCallback):
    r"""A classification loss between the action value net and the target q.

    The target q values are not calculated here and need to be added to
    the state before the loss of this module runs.

    :param name (str): The name of the module. Used e.g. while logging.
    :param q (torch.nn.Module): A deep neural net that outputs the
        discounted loss given the current observations and a given
        action.
    :param opt (torch.optim.Optimizer): An optimizer for q.
    :param lr_schedule (torch.optim.lr_scheduler._LRSchedule): Learning
        rate schedule for the optimizer of q.
    :param max_grad_norm (float): Clip the norm of the gradient during
        backprop using this value.
    :param smoothing_ratio (float): The HL Gauss smoothing ratio is the
        standard deviation of the Gaussian divided by the bin size.
    :param data_group (str): The name of the data group from which this
        Loss takes its data.
    :param log_per_param_weights (bool): If true, log each individual
        policy parameter that is optimized (norm and value histogram).
    :param log_per_param_grads (bool): If true, log the gradients of
        each individual policy parameter that is optimized (norm and
        histogram).
    """

    def __init__(
        self,
        *,
        name: str,
        q: QNet,
        opt: optim.Optimizer,
        lr_schedule: Optional[optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 10.0,
        smoothing_ratio: float = 0.75,
        data_group: str = "default",
        log_per_param_weights=False,
        log_per_param_grads=False,
    ):
        super().__init__(
            name=name,
            optimizer=opt,
            lr_schedule=lr_schedule,
            network=q,
            max_grad_norm=max_grad_norm,
            data_group=data_group,
            log_per_param_weights=log_per_param_weights,
            log_per_param_grads=log_per_param_grads,
        )
        self.q_network = q
        sigma = smoothing_ratio * (q.max_value - q.min_value) / q.num_bins
        loss = HLGaussLoss(q.min_value, q.max_value, q.num_bins, sigma)
        self.hl_gauss = loss.to(q.centers.device)

    def loss(self, observation, actions, q_target):
        logits = self.q_network.logit_net(actions, **observation)
        q_value = self.q_network.q_from_logit(logits)
        self.log_scalar(f"training/{self.name}_prediction", torch.mean(q_value))
        q_target = torch.clamp(q_target, self.q_network.min_value, self.q_network.max_value)
        return self.hl_gauss(logits, q_target)
