"""

"""

import logging
import platform
import signal
import threading
import time

from typing import Callable


if platform.system() in ["Linux", "Darwin"]:
    SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]
else:
    SIGNALS = [signal.SIGINT, signal.SIGTERM]


class ExitSignal:
    """Utility module for handling exit signals across a whole application. This
    module does nothing to end execution - it just helps with notification and
    tracking.

    """

    _stop_now = False
    _installed = False

    _callbacks = []
    _timeout_timers = []

    @staticmethod
    def reset():
        ExitSignal._stop_now = False
        ExitSignal._callbacks = []
        ExitSignal._timeout_timers = []

    @staticmethod
    def add_callback(cb: Callable):
        """Add a callback that will be invoked when the exit signals are triggered.

        :param cb: the callback to invoke
        """
        ExitSignal._callbacks.append(cb)

    @staticmethod
    def install():
        """Install the ExitSignal hook. This will listen to SIGINT, SIGTERM and SIGHUP
        on Unix platforms, and the two former on all others. Will warn if called
        twice.

        """
        if not ExitSignal._installed:
            ExitSignal._installed = True
            for signum in SIGNALS:
                signal.signal(signum, ExitSignal.exit_gracefully)

        else:
            logging.warn("exit signal handler installed twice - no bueno")

    @staticmethod
    def exit_gracefully(signum, frame):
        """Exit gracefully. Primarily intended to be called from a signal handler, but
        may also be called from application code. Both signum and frame are ignored.

        :param signum: ignored
        :param frame: ignored"""
        for callback in ExitSignal._callbacks:
            callback()

        ExitSignal._stop_now = True

    @staticmethod
    def triggered() -> bool:
        """Check if the signal has been triggered.

        :returns: true if a signal has been received
        """

        return ExitSignal._stop_now

    @staticmethod
    def wait(timeout: float = 1):
        """Loop and wait for the signal to be triggered.

        :param timeout: Time to sleep between checking the signal"""
        while not ExitSignal.triggered():
            time.sleep(timeout)

    @staticmethod
    def timeout_after(hours: int = 0, minutes: int = 0, seconds: int = 0):
        """Add a timer to send an exit signal after the provided time. The timer cannot
        be cancelled for now.

        :param hours: number of hours until signal
        :param minutes: number of minutes until signal
        :param seconds: number of seconds until signal
        """
        total_seconds = hours * 3600 + minutes * 60 + seconds
        assert total_seconds > 0, "cannot timeout immediately"
        timer = threading.Timer(total_seconds, ExitSignal.exit_gracefully, args=[0, 1])
        timer.start()
        ExitSignal._timeout_timers.append(timer)
