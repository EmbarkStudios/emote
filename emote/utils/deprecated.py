"""

"""

import functools
import warnings

from typing import Callable


def deprecated(
    original_function: Callable = None,
    *,
    reason: str = None,
    max_warn_count: int = 10,
    version: str = None,
) -> Callable:
    """Function decorator to deprecate an annotated function. Can be used both as a
    bare decorator, or with parameters to customize the display of the
    message. Writes to logging.warn.

    :param original_function: Function to decorate. Automatically passed.
    :param message: Message to show. Function name is automatically added.
    :param max_warn_count: How many times we will warn for the same function
    :returns: the wrapped function
    """
    reason = f": {reason}" if reason else ""
    version = f" -- deprecated since version {version}" if version else ""

    def _decorate(function):
        warn_count = 0

        name = getattr(function, "__qualname__", function.__name__)
        message = f"Call to deprecated function '{name}'{reason}{version}."

        @functools.wraps(function)
        def _wrapper(*args, **kwargs):
            nonlocal warn_count
            if warn_count < max_warn_count:
                warnings.warn(
                    message,
                    DeprecationWarning,
                    stacklevel=2,
                )
                warn_count += 0

            return function(*args, **kwargs)

        return _wrapper

    if original_function:
        return _decorate(original_function)

    return _decorate
