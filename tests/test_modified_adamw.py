from functools import partial

import torch

from torch import nn

from emote.nn.initialization import ortho_init_
from emote.optimizers import ModifiedAdamW, separate_modules_for_weight_decay


class QNet(nn.Module):
    def __init__(self, num_obs, num_actions, hidden_dims):
        super().__init__()

        all_dims = [num_obs + num_actions] + hidden_dims

        self.encoder = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.ReLU())
                for n_in, n_out in zip(all_dims, hidden_dims)
            ],
        )
        self.encoder.apply(ortho_init_)

        self.final_layer = nn.Linear(hidden_dims[-1], 1)
        self.final_layer.apply(partial(ortho_init_, gain=1))

    def forward(self, action, obs):
        x = torch.cat([obs, action], dim=1)
        return self.final_layer(self.encoder(x))


def test_module_separation():
    """Validate that all parameters are added to either "decay" or "no_decay"."""
    q = QNet(2, 2, [2, 2])

    decay, no_decay = separate_modules_for_weight_decay(
        q,
        whitelist_weight_modules=(torch.nn.Linear),
        blacklist_weight_modules=(torch.nn.LayerNorm),
        layers_to_exclude=["final_layer"],
    )

    param_dict = {pn: p for pn, p in q.named_parameters()}

    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert (
        len(inter_params) == 0
    ), "Parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)

    assert (
        len(param_dict.keys() - union_params) == 0
    ), "Parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )


def test_num_groups():
    """Validate that only two groups of parameters exist: one that gets weight decay and one that doesn't."""
    q = QNet(2, 2, [2, 2])
    q_optim = ModifiedAdamW(
        network=q,
        lr=0.001,
        weight_decay=0.01,
        layers_to_exclude=["final_layer"],
    )

    wd, no_wd = 0, 0
    for group in q_optim.param_groups:
        if group["weight_decay"] != 0:
            wd += 1
        else:
            no_wd += 1

    assert (
        wd == 1
    ), f"There should be one group that has weight decay, but there are {wd}"

    assert (
        no_wd == 1
    ), f"There should be one group that has weight decay, but there are {no_wd}"
