"""shoggoth is a torch implementation of embarkrl"""

from .callback import Callback
from .trainer import Trainer, WeakReference
from .sac import QLoss, QTarget, PolicyLoss, AlphaLoss
from . import nn
from . import utils

__all__ = [
    "Callback",
    "Trainer",
    "WeakReference",
    "QLoss",
    "QTarget",
    "PolicyLoss",
    "AlphaLoss",
    "nn",
    "utils",
]

__version__ = "0.1.0"
