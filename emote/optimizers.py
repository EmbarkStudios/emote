# Adapted from
# https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
# (MIT license)

from typing import List, Optional, Set, Tuple

import torch


def separate_modules_for_weight_decay(
    network: torch.nn.Module,
    whitelist_weight_modules: Tuple[torch.nn.Module],
    blacklist_weight_modules: Tuple[torch.nn.Module],
    layers_to_exclude: Optional[List[str]] = None,
) -> Tuple[Set, Set]:
    """Separate the parameters of network into two sets: one set of parameters that will have weight decay, and one set that will not.

    Args:
        network (torch.nn.Module): Network whose modules we want to separate.
        whitelist_weight_modules (Tuple[torch.nn.Module]): Modules that should have weight decay applied to the weights.
        blacklist_weight_modules (Tuple[torch.nn.Module]): Modules that should not have weight decay applied to the weights.
        layers_to_exclude (Optional[List[str]], optional): Names of layers that should be excluded. Defaults to None.

    Returns:
        Tuple(Set, Set): Sets of modules with and without weight decay.
    """

    decay = set()
    no_decay = set()

    for mn, m in network.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn

            if layers_to_exclude is not None and mn in layers_to_exclude:
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

    Arguments:
        :param network: network
        :param learning_rate: learning rate
        :param weight_decay: weight decay coefficient
        :param whitelist_weight_modules: params to get weight decay
            (default: (torch.nn.Linear))
        :param blacklist_weight_modules: params to not get weight decay
            (default: (torch.nn.LayerNorm))
        :param layers_to_exclude: list of names of additional layers to exclude, e.g. last layer of Q-network
            (default: None)
    """

    def __init__(
        self,
        network,
        lr,
        weight_decay,
        whitelist_weight_modules: Tuple[torch.nn.Module] = (torch.nn.Linear),
        blacklist_weight_modules: Tuple[torch.nn.Module] = (torch.nn.LayerNorm),
        layers_to_exclude: List[str] = None,
    ):
        decay, no_decay = separate_modules_for_weight_decay(
            network,
            whitelist_weight_modules,
            blacklist_weight_modules,
            layers_to_exclude,
        )

        param_dict = {pn: p for pn, p in network.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        super().__init__(optim_groups, lr)
