# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [23.0.0] - 2023-03-03

# Breaking

* The minimum required Python version is now 3.9 (#87)

# Added

* Enable exporting ONNX policies for Gaussian MLPs in `emote.extra.onnx_exporter`. This allows you to peridiocally write ONNX files to disk. (#80)
* Add system performance logger in `emote.extra.system_logger`. This'll log memory and CPU usage to Tensorboard. (#81)
* Add memory warmup waiter in `emote.memory.memory` to ensure the memory has enough data before starting to sample. This avoids the collector having to block the training loop when training async. (#78)

# Changed

* The `torch-cpu` feature has been renamed to `torch` as it wasn't limited to CPU-only variants. (#76)
* Our PDM plugin for torch management has been split off into a [separate repository](https://github.com/EmbarkStudios/pdm-plugin-torch/) and [published to PYPI](https://pypi.org/project/pdm-plugin-torch/). (#88)
* Switch to PDM 2.3 as default version for testing (#62)
* The input key used for Feature Agent Proxies can now  (#79)

## [22.0.0] - 2022-10-28

This is the initial release

[23.0.0]: https://github.com/EmbarkStudios/emote/releases/tag/23.0.0
[22.0.0]: https://github.com/EmbarkStudios/emote/releases/tag/22.0.0
