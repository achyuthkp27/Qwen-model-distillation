"""Reproducibility helpers.

Seeds every RNG we might touch. ML backends are seeded opportunistically so this
module imports cleanly even when neither MLX nor PyTorch is installed.
"""

from __future__ import annotations

import os
import random

from distillation.utils.logging import get_logger

_logger = get_logger(__name__)


def seed_everything(seed: int) -> int:
    """Seed all available RNGs for reproducible runs.

    Args:
        seed: The integer seed to apply.

    Returns:
        The same seed, for convenient logging/chaining.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:  # NumPy is a transitive dep of most backends but not guaranteed.
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - optional.
        pass

    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except ImportError:  # pragma: no cover - optional backend.
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:  # pragma: no cover - optional backend.
        pass

    _logger.debug("Seeded all RNGs with %d", seed)
    return seed
