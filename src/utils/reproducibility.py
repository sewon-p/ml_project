"""Seed management for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set seeds for all sources of randomness and enable deterministic behaviour.

    Covers: Python ``random``, ``numpy``, ``torch`` (CPU + CUDA), ``PYTHONHASHSEED``,
    and cuDNN deterministic / benchmark flags.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
