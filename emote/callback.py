from __future__ import annotations

import inspect
import logging
import warnings

from abc import ABCMeta
from functools import wraps
from typing import Any, Dict


def _get_complex(obj, func, arg_names):
    keys_from_member = getattr(func, "__keys_from_members__", {})

    complex_kwargs = {}
    if keys_from_member:
        for arg_name, key in keys_from_member.items():
            key_value = getattr(obj, key)
            del arg_names[arg_names.index(arg_name)]
            complex_kwargs[arg_name] = key_value

    return complex_kwargs


def _make_proxy(func):
    @wraps(func)
    def _proxy(*args, **kwargs):  # noqa pylint: disable=unused-argument
        return func()

    return _proxy


def _make_no_group(func, arg_names, complex_kwargs):
    @wraps(func)
    def _inner_no_group(*args, **kwargs):
        arg_names_ = arg_names[len(args) :]
        kwargs_ = {v: kwargs[v] for v in arg_names_ if v in kwargs}
        for arg_name, key in complex_kwargs.items():
            if key not in kwargs:
                continue

            kwargs_[arg_name] = kwargs[key]

        return func(*args, **kwargs_)

    return _inner_no_group


def _make_group_unpack(func, group, arg_names, complex_kwargs):
    @wraps(func)
    def _inner_fixed_group(*args, **kwargs):
        arg_names_ = arg_names[len(args) :]

        group_ = kwargs[group]
        inner_args = {v: group_[v] for v in arg_names_ if v in group_}
        outer_args = {
            v: kwargs[v] for v in arg_names_ if v not in group_ and v in kwargs
        }
        for arg_name, key in complex_kwargs.items():
            if key in group_:
                inner_args[arg_name] = kwargs[key]

            elif key in kwargs:
                outer_args[arg_name] = kwargs[key]

        res = func(*args, **inner_args, **outer_args)
        if isinstance(res, dict) and group in res:
            group_.update(res[group])
            del res[group]

        return res

    return _inner_fixed_group


def _wrap_callback_function(obj, func, *, group: str = None, use_group: bool = True):
    args = inspect.getfullargspec(func)
    # backward needs to pass things to loss so treated specially.
    # TODO(singhblom) Figure out if this is the nicest way to do it.
    if args.varargs or args.varkw:
        if func.__name__ != "backward" and func.__name__ != "begin_batch":
            warnings.warn(
                f"Deprecated: {func.__qualname__} uses *args or **kwargs, this is deprecated",
                UserWarning,
            )
        return func

    arg_names = args.args + args.kwonlyargs
    if arg_names == ["self"]:
        return _make_proxy(func)

    complex_kwargs = _get_complex(obj, func, arg_names)

    if not use_group:
        return _make_no_group(func, arg_names, complex_kwargs)
    return _make_group_unpack(func, group, arg_names, complex_kwargs)


class CallbackMeta(ABCMeta):
    """The CallbackMeta metaclass modifies the callbacks so that they accept data groups."""

    def __init__(self, cls, bases, fields):
        self._callbacks = {}

        if cls == "Callback":
            self._callbacks = {
                field: method
                for field, method in fields.items()
                if inspect.isfunction(method)
                and field
                in [
                    "begin_training",
                    "begin_cycle",
                    "begin_batch",
                    "backward",
                    "end_batch",
                    "end_cycle",
                    "end_training",
                ]  # TODO(singhblom) Should we filter out unused methods here as well?
            }

        else:
            for base in bases:
                self._callbacks = {**self._callbacks, **getattr(base, "_callbacks", {})}

        for name, func in fields.items():
            if getattr(func, "__is_callback__", None):
                self._callbacks[name] = func

        super().__init__(cls, bases, fields)

    def __call__(self, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)

        for name, func in self._callbacks.items():
            concrete_func = getattr(instance, name)

            if concrete_func.__qualname__ == func.__qualname__:
                logging.debug("skipping patch of %s: not overridden", func.__qualname__)
                continue

            group_name = getattr(instance, "data_group", None)
            if not group_name:
                group_name = getattr(instance.__class__, "DATA_GROUP", None)

            concrete_func = _wrap_callback_function(
                instance,
                concrete_func,
                group=group_name,
                use_group=group_name is not None,
            )
            setattr(instance, name, concrete_func)

        return instance

    def extend(self, func):
        func.__is_callback__ = True
        return func

    def keys_from_member(self, **arg_name_to_key):
        def _wrap(func):
            func.__keys_from_members__ = arg_name_to_key
            return func

        return _wrap


class Callback(metaclass=CallbackMeta):
    """The principal modular building block of emote.

    Callbacks are modular pieces of code that together build up the training loop.
    They contain hooks that are executed at different points during training.
    These can consume values from other callbacks, and generate their own for others
    to consume. This allows a very loosely coupled flow of data between different
    parts of the code. The most important examples of callbacks in emote are the
    Losses.

    The concept has been borrowed from Keras and FastAI.
    """

    def __init__(self, cycle: int | None = None):
        super().__init__()
        self._order = 0
        self.cycle = cycle

    def begin_training(self, *args, **kwargs):
        """Called when training starts, both from scratch and when restoring
        from a checkpoint."""
        pass

    def begin_cycle(self, *args, **kwargs):
        """Called at the start of each cycle."""
        pass

    def begin_batch(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass

    def end_batch(self, *args, **kwargs):
        pass

    def end_cycle(self, *args, **kwargs):
        pass

    def end_training(self, *args, **kwargs):
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass
