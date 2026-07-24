"""Student model backends behind a common interface (Phase 3).

MLX is the Apple-Silicon-native default; PEFT (PyTorch) is the alternative. Both
implement :class:`StudentBackend`. Concrete backends are imported lazily via
:func:`get_backend` so this package imports without any ML framework installed.
"""

from __future__ import annotations

from distillation.student.base import (
    EncodedSample,
    GenerationResult,
    StepMetrics,
    StudentBackend,
)
from distillation.student.factory import get_backend

__all__ = [
    "EncodedSample",
    "GenerationResult",
    "StepMetrics",
    "StudentBackend",
    "get_backend",
]
