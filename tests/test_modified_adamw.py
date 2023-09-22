from functools import partial

import pytest
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


def module_separation(param_dict, decay, no_decay):
    inter_params = decay & no_decay
    union_params = decay | no_decay

    assert (
        len(inter_params) == 0
    ), f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"

    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"Parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"


def num_groups(param_groups):
    wd, no_wd = 0, 0
    for group in param_groups:
        if group["weight_decay"] != 0:
            wd += 1
        else:
            no_wd += 1

    assert wd == 1, f"There should be one group that has weight decay, but there are {wd}"

    assert no_wd == 1, f"There should be one group that has no weight decay, but there are {no_wd}"


def test_module_separation():
    """Validate that all parameters are added to either "decay" or "no_decay"."""
    q = QNet(2, 2, [2, 2])

    decay, no_decay = separate_modules_for_weight_decay(
        q,
        whitelist_weight_modules=(torch.nn.Linear,),
        blacklist_weight_modules=(torch.nn.LayerNorm,),
        layers_to_exclude={"final_layer"},
    )

    param_dict = dict(q.named_parameters())

    module_separation(param_dict, decay, no_decay)

    # Make sure the test fails when a module is not added to either of the sets
    with pytest.raises(
        AssertionError,
        match="Parameters {'test'} were not separated into either decay/no_decay set!",
    ):
        param_dict["test"] = None
        module_separation(param_dict, decay, no_decay)

    # Make sure the test fails when a module is added to both of the sets
    with pytest.raises(
        AssertionError,
        match="Parameters {'test'} made it into both decay/no_decay sets!",
    ):
        decay.add("test")
        no_decay.add("test")

        module_separation(param_dict, decay, no_decay)


def test_num_groups():
    """Validate that only two groups of parameters exist: one that gets weight decay and one that doesn't."""
    q = QNet(2, 2, [2, 2])

    q_optim = ModifiedAdamW(
        network=q,
        lr=0.001,
        weight_decay=0.01,
        layers_to_exclude=["final_layer"],
    )

    num_groups(q_optim.param_groups)

    # Make sure the test fails when a group with weight decay doesn't exist
    with pytest.raises(
        AssertionError,
        match="There should be one group that has weight decay, but there are 0",
    ):
        for i in range(len(q_optim.param_groups)):
            if q_optim.param_groups[i]["weight_decay"] > 0:
                del q_optim.param_groups[i]
                break

        num_groups(q_optim.param_groups)

    # Make sure the test fails when multiple groups with weight decay exist
    with pytest.raises(
        AssertionError,
        match="There should be one group that has weight decay, but there are 2",
    ):
        q_optim.param_groups += [{"weight_decay": 0.1}, {"weight_decay": 0.2}]

        num_groups(q_optim.param_groups)
