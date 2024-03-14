"""Emote is a torch implementation of embarkrl.

Emote
========


In order to do reinforcement learning we need to have two things:
A **learning protocol** that specifies which losses to use, which network
architectures, which optimizers, and so forth. We also need some kind of
**data collector** that interacts with the world and stores the experiences
from that in a way which makes them accessible to the learning protocol.

In Emote, data collection is done by Collectors, the protocol for the
learning algorithm is built up of Callbacks, and they are tied together
by a Trainer.
"""

from . import nn, utils
from .algorithms import sac
from .callback import Callback
from .trainer import Trainer, WeakReference


__all__ = [
    "Callback",
    "Trainer",
    "WeakReference",
    "sac",
    "nn",
    "utils",
]

__version__ = "0.1.0"
