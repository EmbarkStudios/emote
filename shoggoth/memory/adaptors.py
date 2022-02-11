"""

"""

from typing import List
from .memory import Table
import torch


def DictObsAdaptor(
    memory: Table,
    keys: List[str],
    output_keys: List[str] = None,
    with_next: bool = True,
):
    """
    Converts multiple observation columns to a single dict observation.

    :param memory: The table to adapt
    :param keys: The dictionary keys to extract
    :param output_keys: The output names for the extracted keys. Defaults to the same name.
    """
    assert (
        "dict_obs_adaptor" not in memory.unique_adaptors
    ), "only one dict adaptor can be used per memory"
    memory.unique_adaptors.add("dict_obs_adaptor")
    original_sample = memory.sample

    if output_keys is None:
        output_keys = keys
    else:
        assert len(keys) == len(output_keys)

    def _sample(point, length):
        result = original_sample(point, length)

        obs_dict = {}
        next_obs_dict = {}
        for (key, out_key) in zip(keys, output_keys):
            obs_dict[out_key] = result.pop(key)
            if with_next:
                next_obs_dict[f"{out_key}"] = result.pop("next_" + key)

        result["observation"] = obs_dict
        result["next_observation"] = next_obs_dict
        return result

    memory.sample = _sample
    return memory


def IntrinsicRewardAdaptor(
    memory: Table,
    pure: bool,
    reward_key: str = "rewards",
    intrinsic_reward_key: str = "intrinsic_rewards",
):
    """The intrinsic reward adaptor can be used to sum two sampled arrays elementwise.

    :param memory: The table to adapt
    :param pure: Whether to sum or replace the original reward
    :param reward_key: The reward key to update
    :param intrinsic_reward_key: The key containing the intrinsic reward to read from"""
    assert (
        "intrinsic_reward_adaptor" not in memory.unique_adaptors
    ), "only one intrinsic reward adaptor can be used per memory"
    memory.unique_adaptors.add("intrinsic_reward_adaptor")
    original_sample = memory.sample

    if pure:

        def _sample(point, length):
            result = original_sample(point, length)
            result[reward_key] = result[intrinsic_reward_key]
            return result

    else:

        def _sample(point, length):
            result = original_sample(point, length)
            result[reward_key] = result[reward_key] + result[intrinsic_reward_key]
            return result

    memory.sample = _sample
    return memory


def KeyScaleAdaptor(memory: Table, key: str, scale: float):
    """An adaptor to apply scaling to a specified sampled key.

    :param memory: The table to adapt
    :param key: The key for which to scale data
    :param scale: The scale factor to apply

    """
    scale = torch.tensor(scale)
    original_sample = memory.sample

    def _sample(point, length):
        result = original_sample(point, length)
        result[key] *= scale
        return result

    memory.sample = _sample
    return memory


def KeyCastAdaptor(memory: Table, key: str, dtype: type):
    """An adaptor to cast a specified sampled key.

    :param memory: The table to adapt
    :param key: The key for which to cast data
    :param scale: The scale factor to apply
    """
    original_sample = memory.sample

    def _sample(point, length):
        result = original_sample(point, length).type(dtype)
        return result

    memory.sample = _sample
    return memory


def TerminalAdaptor(memory: Table, value_key: str, target_key: str) -> Table:
    """An adaptor to apply tags from detailed terminal tagging.

    :param memory: The table to adapt
    :param value_key: the key containing the terminal mask value to apply
    :param target_key: the default mask data to override
    """
    assert (
        "terminal_adaptor" not in memory.unique_adaptors
    ), "only one terminal adaptor can be used per memory"
    memory.unique_adaptors.add("terminal_adaptor")

    original_sample = memory.sample

    # Note: The below code assumes that both terminal tags and the masks are
    # always 1.0 or 0.0.
    def _sample(point, length):
        result = original_sample(point, length)

        # reshapes the input data to [batch_size, time_dimension, ...]  so we
        # can correctly overlay the terminal tags - otherwise we get a dimension
        # mismatch.
        result_shape = result[target_key].shape
        new_result_shape = (-1, length, *result_shape[1:])
        result_reshaped = torch.reshape(result[target_key], new_result_shape)

        # compute a selection mask which is true for every non-end-of-episode step
        indice_mask = result_reshaped == 1.0
        # where it is true, simply use the existing value in
        # result[target_key], otherwise use the terminal-state tag value
        # from result[value_key]
        result_reshaped = torch.where(
            indice_mask,
            result_reshaped,
            torch.unsqueeze(torch.unsqueeze(result[value_key], -1), -1),
        )

        # reshape back
        result[target_key] = torch.reshape(result_reshaped, result_shape)
        return result

    memory.sample = _sample
    return memory
