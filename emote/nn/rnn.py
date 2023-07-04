from __future__ import annotations

import numpy as np
import torch

from torch import Tensor, nn

from emote.callback import Callback
from emote.memory.core_types import SampleResult
from emote.mixins.logging import LoggingMixin
from emote.nn.initialization import ortho_init_
from emote.utils.timed_call import BlockTimers


class BurnInSamplerAdaptor:
    """This adaptor splits the sampled data into a burn in part and a train part.

    ... warning:

        The burn in data is returned as PaddedSequence, not a regular Tensor. Special care must be
        taken when using this type. Note that it stores data internally as [L, B, *], while most of
        Emote assumes [B * L, *].

    The burn in part is intended for use with a BurnInCallback to update the hidden states before
    the backprop. This should help with the stability of the training, as the hidden states can
    otherwise drift away from the current meaning. The method here is derived from the R2D2
    algorithm.

    An implementation detail worth noting is that when the sampling offset is shorter than the burn
    in, the burn in part will be shortened and padded with zeros. This ensures that the sampling
    probabilities of the underlying sequence are respected. Special care might need to be taken with
    a prioritized sampler as the probability used would be for where the burn in begins, not the
    training data.

    The above issue could be solved by invoking the sample method twice, but since we couldn't hold
    the lock between the samples we would be at risk of data corruption! Thus, it'd need a bigger
    refactor.

    """

    def __init__(
        self,
        keys: list[str],
        packed_keys: list[str],
        burn_in_length: int,
    ):
        self.burn_in_length = burn_in_length
        self.keys = keys
        self.packed_keys = packed_keys

    def __call__(
        self, result: SampleResult, count: int, sequence_length: int
    ) -> SampleResult:
        sample_points = result["sample_points"]
        burn_in_lengths = np.zeros(count)
        burn_in_masks = np.zeros(count)

        new_sample_points = []
        for offset, (episode, start, end) in enumerate(sample_points):
            resampled_start = max(start - self.burn_in_length, 0)
            resampled_end = start

            burn_in_lengths[offset] = min(start, self.burn_in_length)
            burn_in_masks[offset] = 1.0 if burn_in_lengths[offset] > 0 else 0
            new_sample_points.append((episode, resampled_start, resampled_end))

        sampled_data = self._table._execute_gather(
            len(new_sample_points), self.burn_in_length, new_sample_points
        )

        for key in self.keys:
            result[f"burn_in_{key}"] = sampled_data[key]

        result["burn_in_lengths"] = torch.from_numpy(burn_in_lengths)
        result["burn_in_length"] = result["burn_in_lengths"].to(torch.int64)
        result["burn_in_masks"] = torch.from_numpy(burn_in_masks)

        return result


class BurnInDictObsAdaptor:
    """
    Converts multiple observation columns to a single dict observation.

    :param keys: The dictionary keys to extract
    :param output_keys: The output names for the extracted keys. Defaults to the same
        name.
    :param with_next: If True, adds an extra column called "next_{key}" for each key
        in keys.
    """

    def __init__(
        self,
        keys: list[str],
        output_keys: list[str] | None = None,
        prefix: str = "burn_in",
    ):
        if output_keys is None:
            output_keys = keys
        else:
            assert len(keys) == len(output_keys)
        self.key_map = list(zip(keys, output_keys))
        self.prefix = prefix

    def __call__(
        self, result: SampleResult, count: int, sequence_length: int
    ) -> SampleResult:
        burn_in_obs_dict = {}
        for key, out_key in self.key_map:
            burn_in_obs_dict[out_key] = result.pop(f"{self.prefix}_{key}")

        result[f"{self.prefix}_observation"] = burn_in_obs_dict

        return result


class BurnInCallback(Callback, LoggingMixin):
    """Implements a burn-in method derived from the R2D2 algorithm.

    The burn-in method is used to update the hidden states of the policy before the
    backpropagation. This is achieved by running the policy on the burn-in data and then using
    the hidden states as the initial hidden states for the backpropagation.
    """

    def __init__(
        self,
        input_keys: list[str],
        output_key: str,
        policy: nn.Module,
        burn_in_length: int,
        data_group: str = "default",
    ):
        """Creates a new BurnInCallback for updating hidden states.

        :param input_keys: The keys to extract from the observation and pass to the policy.
        :param output_key: The key to extract from the observation and use as the target.
        :param policy: The policy to run the burn-in on.
        :param burn_in_length: The length of the burn-in.
        :param data_group: The data group to apply the callback to.
        """
        super().__init__()
        self.keys = input_keys
        self.output_key = output_key
        self.policy = policy
        self.burn_in_length = burn_in_length
        self.data_group = data_group

        self._order = -10
        self._timers = BlockTimers()

    def backward(
        self,
        observation: dict[str, Tensor],
        burn_in_observation: dict[str, Tensor],
        burn_in_masks: Tensor,
        burn_in_lengths: Tensor,
        batch_size: int,
        rollout_length: int,
    ):
        burn_in_data = tuple(burn_in_observation[key] for key in self.keys)

        with torch.no_grad():
            with self._timers.scope("forward"):
                _, state = self.policy(
                    *burn_in_data,
                    rollout_length=self.burn_in_length,
                    padding_lengths=burn_in_lengths,
                )

            with self._timers.scope("mask"):
                assign_target = observation[self.output_key].view(
                    -1, rollout_length, *observation[self.output_key].shape[1:]
                )

                burn_in_masks = torch.unsqueeze(burn_in_masks, -1)

                # The mask is 0 if we should use the "real" state and 1 if we should use the new
                # state. This ensures we get to see initial state for state 0.

                original_state = assign_target[:, 0]
                new_state = state * burn_in_masks
                old_state = original_state * (1 - burn_in_masks)

                assign_target[:, 0] = new_state + old_state

                observation[self.output_key] = assign_target.reshape(
                    -1, *assign_target.shape[2:]
                )

            h_t_shift = torch.nn.functional.kl_div(
                state,
                original_state,
                reduction="batchmean",
            )

        self.log_windowed_scalar("burn_in/divergence", h_t_shift.item())
        self.log_histogram("burn_in/masks", burn_in_masks)
        self.log_histogram("burn_in/lengths", burn_in_lengths)

        for name, (mean, _) in self._timers.stats().items():
            self.log_scalar(f"burn_in/{name}", mean)


class GruEncoder(nn.Module):
    """GRU encoder module with an optional input encoder."""

    def __init__(
        self,
        encoder_dim: int,
        gru_hidden: int,
        batch_size: int,
        rollout_length: int,
        input_encoder: nn.Module | None = None,
    ):
        """Creates a new GRU encoder.

        :param encoder_dim: The dimension of the input to the GRU - either the feature size or the
                            output of the optional encoder.
        :param gru_hidden: The hidden dimension of the GRU.
        :param batch_size: The batch size.
        :param rollout_length: The rollout length.
        :param input_encoder: An optional encoder to apply to the input before passing it to the GRU
        """
        super().__init__()
        self.encoder = input_encoder
        self.gru = nn.GRU(encoder_dim, gru_hidden, batch_first=True)
        self.gru.apply(ortho_init_)

        self.output_dim = gru_hidden
        self.encoder_dim = encoder_dim
        self.rollout_length = rollout_length
        self.batch_size = batch_size

    def forward(
        self,
        obs: Tensor,
        gru_hidden: Tensor,
        rollout_length: int | None = None,
        padding_lengths: Tensor | None = None,
    ) -> Tensor:
        """Forward pass of the GRU encoder."""

        if self.encoder is not None:
            encoded = self.encoder(obs)

        else:
            encoded = obs

        if (
            rollout_length is not None
            or obs.shape[0] == self.batch_size * self.rollout_length
        ):
            rollout_length = rollout_length or self.rollout_length
            encoded = encoded.view(-1, rollout_length, self.encoder_dim)

            if padding_lengths is not None:
                encoded = torch.nn.utils.rnn.pack_padded_sequence(
                    encoded,
                    np.maximum(padding_lengths, 1),
                    batch_first=True,
                    enforce_sorted=False,
                )

            gru_hidden = gru_hidden.view(-1, rollout_length, self.output_dim)[
                :, 0:1, :
            ].transpose(0, 1)

        else:
            encoded = encoded.view(obs.shape[0], 1, -1)
            gru_hidden = gru_hidden.view(obs.shape[0], 1, -1).transpose(0, 1)

        encoded, gru_hidden = self.gru(encoded, gru_hidden)

        if padding_lengths is not None:
            encoded = torch.vstack(torch.nn.utils.rnn.unpack_sequence(encoded))
        else:
            encoded = encoded.reshape(-1, self.output_dim)

        gru_hidden = gru_hidden.view(-1, self.output_dim)

        return encoded, gru_hidden
