"""shoggoth is a torch implementation of embarkrl

Shoggoth
========


In order to do reinforcement learning we need to have two things: 
A **learning protocol** that specifies which losses to use, which network
architectures, which optimizers, and so forth. We also need some kind of
**data collector** that interacts with the world and stores the experiences
from that in a way which makes them accessible to the learning protocol.

In Shoggoth, data collection is done by Collectors, the protocol for the
learning algorithm is built up of Callbacks, and they are tied together
by a Trainer.

"""

from .callback import Callback
from .callbacks import LossCallback, LoggingCallback
from .trainer import Trainer, WeakReference
from . import sac
from . import nn
from . import utils

__all__ = [
    "Callback",
    "LossCallback",
    "LoggingCallback",
    "Trainer",
    "WeakReference",
    "sac",
    "nn",
    "utils",
]

__version__ = "0.1.0"
