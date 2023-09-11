# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Features

- `OnnxExporter.add_metadata` allows setting metadata to export with the policy. The key has to be a string, and the value has to be convertible to string.

### Changes

#### New memory serialization format.
A new version of memory Table export has been introduced. This removes the need for cloudpickle,
while focusing on restoring data. Old memories can still be imported, and you can also force the old
format by passing `version = TableSerializationFormat.Legacy` in `Table.store`. The new format works
by simply ingesting all data from the memory using the regular `add_sequence`, instead of filling
the data stores directly. As part of this, the `TableArray.store` and `TableArray.restore` functions
have new arguments to handle versioning.

  - New functions:
    - `Column.configuration` and `Column.configure` for save and load respectively
	- `Strategy.clear` to clear all state
	- `Strategy.state` and `Strategy.load_state` for save and load of strategy data.
	- `Strategy.begin_simple_import` and `Strategy.end_simple_import` to bookend the import process.
	- `Strategy._in_simple_import` to allow derived classes to bypass work while import is happening.

#### Other changes:

  - Now targetting torch version 1.12, up from 1.11.
  - `OnnxExporter` accepts a `device` argument to enable tracing on other devices.
  - `FinalRewardTestCheck` can now be configured with another key and to use windowed data.

### Deprecations

- `emote.callbacks` has been converted to a package. Future built-in
  callbacks will not be re-exported from `emote.callbacks`, and should
  instead be imported from their internal location.
- `emote.callbacks.LoggingMixin` is now in the `emote.mixins.logging` module instead.

### Bugfixes

- Fix `FeatureAgentProxy.input_names` to use `input_key` when configured.
- `Callback.cycle` can now be `None`
- Fixed a deprecation warning with `np.bool_` being used.

## [23.0.0] - 2023-03-03

### Breaking

* The minimum required Python version is now 3.9 (#87)
* The `torch-cpu` feature has been renamed to `torch` as it wasn't limited to CPU-only variants. (#76)

### Added

* Enable exporting ONNX policies for Gaussian MLPs in `emote.extra.onnx_exporter`. This allows you to peridiocally write ONNX files to disk. (#80)
* Add system performance logger in `emote.extra.system_logger`. This'll log memory and CPU usage to Tensorboard. (#81)
* Add memory warmup waiter in `emote.memory.memory` to ensure the memory has enough data before starting to sample. This avoids the collector having to block the training loop when training async. (#78)

### Changed

* Our PDM plugin for torch management has been split off into a [separate repository](https://github.com/EmbarkStudios/pdm-plugin-torch/) and [published to PYPI](https://pypi.org/project/pdm-plugin-torch/). (#88)
* Switch to PDM 2.3 as default version for testing (#62)
* The input key used for Feature Agent Proxies can now be customized (#79)

## [22.0.0] - 2022-10-28

This is the initial release

[Unreleased]: https://github.com/EmbarkStudios/emote/compare/v23.0.0...HEAD
[23.0.0]: https://github.com/EmbarkStudios/emote/releases/tag/v23.0.0
[22.0.0]: https://github.com/EmbarkStudios/emote/releases/tag/v22.0.0
