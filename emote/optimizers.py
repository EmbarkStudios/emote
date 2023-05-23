# Adapted from
# https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
# (MIT license)

from __future__ import annotations

from typing import Type

import torch


def separate_modules_for_weight_decay(
    network: torch.nn.Module,
    whitelist_weight_modules: tuple[Type[torch.nn.Module], ...],
    blacklist_weight_modules: tuple[Type[torch.nn.Module], ...],
    layers_to_exclude: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Separate the parameters of network into two sets: one set of parameters that will have weight decay, and one set that will not.

    Args:
        network (torch.nn.Module): Network whose modules we want to separate.
        whitelist_weight_modules (tuple[Type[torch.nn.Module], ...]): Modules that should have weight decay applied to the weights.
        blacklist_weight_modules (tuple[Type[torch.nn.Module], ...]): Modules that should not have weight decay applied to the weights.
        layers_to_exclude (set[str] | None, optional): Names of layers that should be excluded. Defaults to None.

    Returns:
        tuple[set[str], set[str]]: Sets of modules with and without weight decay.
    """
    # Make sure the same module doesn't appear in both whitelist_weight_modules and blacklist_weight_modules
    assert (
        len(set(whitelist_weight_modules) & set(blacklist_weight_modules)) == 0
    ), "Some modules are both whitelisted and blacklisted!"

    layers_to_exclude = layers_to_exclude or set()
    decay = set()
    no_decay = set()

    for mn, m in network.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn

            if mn in layers_to_exclude:
                # Weights of excluded layers will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith("bias"):
                # Biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # Weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # Weights of whitelist modules will be weight decayed
                decay.add(fpn)

    return decay, no_decay


class ModifiedAdamW(torch.optim.AdamW):
    """Modifies AdamW (Adam with weight decay) to not apply weight decay on the bias and layer normalization weights, and optionally additional modules.

    Args:
        network (torch.nn.Module): network
        lr (float): learning rate
        weight_decay (float): weight decay coefficient
        whitelist_weight_modules (tuple[Type[torch.nn.Module], ...], optional): params to get weight decay. Defaults to (torch.nn.Linear, ).
        blacklist_weight_modules (tuple[Type[torch.nn.Module], ...], optional): params to not get weight decay. Defaults to (torch.nn.LayerNorm, ).
        layers_to_exclude (set[str] | None, optional): set of names of additional layers to exclude, e.g. last layer of Q-network. Defaults to None.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        lr: float,
        weight_decay: float,
        whitelist_weight_modules: tuple[Type[torch.nn.Module], ...] = (
            torch.nn.Linear,
        ),
        blacklist_weight_modules: tuple[Type[torch.nn.Module], ...] = (
            torch.nn.LayerNorm,
        ),
        layers_to_exclude: set[str] | None = None,
    ):
        decay, no_decay = separate_modules_for_weight_decay(
            network,
            whitelist_weight_modules,
            blacklist_weight_modules,
            layers_to_exclude,
        )

        param_dict = dict(network.named_parameters())

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in decay],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in no_decay],
                "weight_decay": 0.0,
            },
        ]

        super().__init__(optim_groups, lr)
