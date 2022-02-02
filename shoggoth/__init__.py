"""shoggoth is a torch implementation of embarkrl"""

from .callback import Callback
from .trainer import Trainer, WeakReference
from .sac import QLoss, QTarget, PolicyLoss, AlphaLoss

__all__ = [
    "Callback",
    "Trainer",
    "WeakReference",
    "QLoss",
    "QTarget",
    "PolicyLoss",
    "AlphaLoss",
]

__version__ = "0.1.0"
