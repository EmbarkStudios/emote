import time

from typing import Dict, List

from emote.callback import Callback
from emote.mixins.logging import LoggingMixin


try:
    import wandb
except ImportError as root:
    raise ImportError(
        "enable the optional `wandb` feature to use the WBLogger"
    ) from root


class WBLogger(Callback):
    """Logs the provided loggable callbacks to Weights&Biases."""

    def __init__(
        self,
        callbacks: List[LoggingMixin],
        config: Dict,
        log_interval: int,
    ):
        super().__init__(cycle=log_interval)

        self._cbs = callbacks
        self._config = config

        assert wandb.run is None
        wandb.init(
            project=self._config["wandb_project"],
            name=self._config["wandb_run"],
            config=wandb.helper.parse_config(
                self._config, exclude=("wandb_project", "wandb_run")
            ),
        )

    def begin_training(self):
        self._start_time = time.monotonic()

    def end_cycle(self, bp_step, bp_samples):
        log_dict = {}
        suffix = "bp_step"

        for cb in self._cbs:
            for k, v in cb.scalar_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                log_dict[k] = v

            for k, v in cb.windowed_scalar.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                log_dict[k] = sum(v) / len(v)

            for k, v in cb.windowed_scalar_cumulative.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                log_dict[f"{k}/cumulative"] = v

            for k, v in cb.image_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)
                log_dict[k] = wandb.Image(v)

            for k, (video_array, fps) in cb.video_logs.items():
                if suffix:
                    k_split = k.split("/")
                    k_split[0] = k_split[0] + "_" + suffix
                    k = "/".join(k_split)

                log_dict[k] = wandb.Video(video_array, fps=fps)

        time_since_start = time.monotonic() - self._start_time
        log_dict["performance/bp_samples_per_sec"] = bp_samples / time_since_start
        log_dict["performance/bp_steps_per_sec"] = bp_step / time_since_start
        log_dict["log/bp_step"] = bp_step
        wandb.log(log_dict)

    def end_training(self):
        wandb.finish()
        return super().end_training()
