""""""

from typing import Callable, List, Optional

import torch

from emote.memory.core_types import SampleResult


Adaptor = Callable[[SampleResult, int, int], SampleResult]


class DictObsAdaptor:
    """Converts multiple observation columns to a single dict observation.

    :param keys: The dictionary keys to extract
    :param output_keys: The output names for the extracted keys.
        Defaults to the same name.
    :param with_next: If True, adds an extra column called "next_{key}"
        for each key in keys.
    """

    def __init__(
        self,
        keys: List[str],
        output_keys: Optional[List[str]] = None,
        with_next: bool = True,
    ):
        if output_keys is None:
            output_keys = keys
        else:
            assert len(keys) == len(output_keys)
        self.key_map = list(zip(keys, output_keys))
        self.with_next = with_next

    def __call__(self, result: SampleResult, count: int, sequence_length: int) -> SampleResult:
        obs_dict = {}
        next_obs_dict = {}
        for key, out_key in self.key_map:
            obs_dict[out_key] = result.pop(key)
            if self.with_next:
                next_obs_dict[f"{out_key}"] = result.pop("next_" + key)

        result["observation"] = obs_dict
        result["next_observation"] = next_obs_dict
        return result


class KeyScaleAdaptor:
    """An adaptor to apply scaling to a specified sampled key.

    :param key: The key for which to scale data
    :param scale: The scale factor to apply
    """

    def __init__(self, scale, key):
        self.key = key
        self.scale = torch.tensor(scale)

    def __call__(self, result: SampleResult, count: int, sequence_length: int) -> SampleResult:
        result[self.key] *= self.scale
        return result


class KeyCastAdaptor:
    """An adaptor to cast a specified sampled key.

    :param key: The key for which to cast data
    :param dtype: The dtype to cast to.
    """

    def __init__(self, dtype, key):
        self.key = key
        self.dtype = dtype

    def __call__(self, result: SampleResult, count: int, sequence_length: int) -> SampleResult:
        result[self.key].to(self.dtype)
        return result


class TerminalAdaptor:
    """An adaptor to apply tags from detailed terminal tagging.

    :param value_key: the key containing the terminal mask value to
        apply
    :param target_key: the default mask data to override
    """

    def __init__(self, target_key: str, value_key: str) -> None:
        self.target_key = target_key
        self.value_key = value_key

    def __call__(self, result: SampleResult, count: int, sequence_length: int) -> SampleResult:
        # Note: The below code assumes that both terminal tags and the masks are
        # always 1.0 or 0.0.

        # reshapes the input data to [batch_size, time_dimension, ...]  so we
        # can correctly overlay the terminal tags - otherwise we get a dimension
        # mismatch.
        target = result[self.target_key]
        value = result[self.value_key]

        result_shape = target.shape

        new_value_shape = (-1, sequence_length, *result_shape[1:])
        value_reshaped = torch.reshape(value, new_value_shape)

        # compute a selection mask which is true for every non-end-of-episode step
        indice_mask = target == 1.0
        # where it is true, simply use the existing value in
        # result[target_key], otherwise use the terminal-state tag value
        # from result[value_key]
        result_reshaped = torch.where(
            indice_mask,
            target,
            value_reshaped[:, -1],
        )

        # reshape back
        result[self.target_key] = torch.reshape(result_reshaped, result_shape)
        return result
