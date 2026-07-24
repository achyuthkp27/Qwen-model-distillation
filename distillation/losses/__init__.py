"""Distillation strategies (Phase 4) and backend-agnostic loss ops.

Only response distillation is implemented today; logit/hidden/attention
strategies plug in via :func:`register_strategy` without touching the trainer.
"""

from __future__ import annotations

from distillation.losses.base import (
    DistillationStrategy,
    LossInputs,
    LossResult,
    get_strategy,
    register_strategy,
)
from distillation.losses.ops import ArrayOps, NumpyOps
from distillation.losses.response import ResponseDistillation

__all__ = [
    "ArrayOps",
    "DistillationStrategy",
    "LossInputs",
    "LossResult",
    "NumpyOps",
    "ResponseDistillation",
    "get_strategy",
    "register_strategy",
]
