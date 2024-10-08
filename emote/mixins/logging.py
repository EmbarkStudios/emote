from collections import deque
from collections.abc import Iterable
from typing import Any, Dict, Tuple

import numpy as np
import torch


class LoggingMixin:
    """A Mixin that accepts logging calls.

    Logged data is saved on this object and gets written by a Logger.
    This therefore doesn't care how the data is logged, it only provides
    a standard interface for storing the data to be handled by a Logger.
    """

    def __init__(self, *, default_window_length: int = 250, **kwargs):
        super().__init__(**kwargs)

        self.scalar_logs: Dict[str, float | torch.Tensor] = {}
        self.windowed_scalar: Dict[str, deque[float | torch.Tensor]] = {}
        self.windowed_scalar_cumulative: Dict[str, int] = {}
        self.image_logs: Dict[str, torch.Tensor] = {}
        self.hist_logs: Dict[str, float | torch.Tensor] = {}
        self.video_logs: Dict[str, Tuple[np.ndarray, int]] = {}

        self._default_window_length = default_window_length

    def log_scalar(self, key: str, value: float | torch.Tensor):
        """Use log_scalar to periodically log scalar data."""
        if isinstance(value, torch.Tensor):
            self.scalar_logs[key] = value.item()
        else:
            self.scalar_logs[key] = value

    def log_windowed_scalar(
        self,
        key: str,
        value: float | torch.Tensor | Iterable[torch.Tensor | float],
    ):
        """Log scalars using a moving window average.

        By default this will use `default_window_length` from the constructor as the window
        length. It can also be overridden on a per-key basis using the format
        windowed[LENGTH]:foo/bar. Note that this cannot be changed between multiple invocations -
        whichever length is found first will be permanent.
        """

        if key not in self.windowed_scalar:
            # we allow windowed[100]:some_key/foobar to override window size
            if "windowed[" in key:
                p = key.split(":")[0]
                length = int(p.split("[")[1][:-1])
            else:
                length = self._default_window_length

            self.windowed_scalar[key] = deque(maxlen=length)
            self.windowed_scalar_cumulative[key] = 0

        if isinstance(value, Iterable):
            val = value.numpy() if isinstance(value, torch.Tensor) else value
            self.windowed_scalar[key].extend(val)
            self.windowed_scalar_cumulative[key] += sum(val)
        else:
            val = value.item() if isinstance(value, torch.Tensor) else value
            self.windowed_scalar[key].append(val)
            self.windowed_scalar_cumulative[key] += val

    def log_image(self, key: str, value: torch.Tensor):
        """Use log_image to periodically log image data."""
        if len(value.shape) == 3:
            self.image_logs[key] = value.detach()

    def log_video(self, key: str, value: Tuple[np.ndarray, int]):
        """Use log_scalar to periodically log scalar data."""
        self.video_logs[key] = value

    def log_histogram(
        self,
        key: str,
        value: torch.Tensor | float | Iterable[torch.Tensor | float],
    ):
        if isinstance(value, Iterable):
            self.hist_logs[key] = value.detach() if isinstance(value, torch.Tensor) else value
        else:
            if key not in self.hist_logs:
                self.hist_logs[key] = deque(maxlen=self._default_window_length)

            self.hist_logs[key].append(value)

    def state_dict(self):
        try:
            state_dict = super().state_dict()
        except AttributeError:
            print(f"Error getting the state dict for {self.__class__.__name__}.")
            print("This sometimes happens when the LoggingMixin is inherited from after")
            print(
                f"the other mixins. The current method resolution order is {self.__class__.__mro__}"
            )
            raise
        state_dict["scalar_logs"] = self.scalar_logs
        state_dict["hist_logs"] = self.hist_logs
        state_dict["image_logs"] = self.image_logs
        state_dict["video_logs"] = self.video_logs
        state_dict["windowed_scalar"] = {
            k: (list(v), v.maxlen) for (k, v) in self.windowed_scalar.items()
        }
        state_dict["windowed_scalar_cumulative"] = self.windowed_scalar_cumulative
        return state_dict

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        load_network: bool = True,
        load_optimizer: bool = True,
        load_hparams: bool = True,
    ):
        if load_hparams:
            self.scalar_logs = state_dict.pop("scalar_logs")
            self.hist_logs = state_dict.pop("hist_logs")
            self.video_logs = state_dict.pop("video_logs")
            self.image_logs = state_dict.pop("image_logs")
            self.windowed_scalar = {
                k: deque(v[0], maxlen=v[1]) for (k, v) in self.windowed_scalar.items()
            }
            self.windowed_scalar_cumulative = state_dict.pop("windowed_scalar_cumulative")

        super().load_state_dict(state_dict, load_network, load_optimizer, load_hparams)
