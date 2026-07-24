"""Student-backend factory (Phase 3).

Selects and constructs the concrete :class:`StudentBackend` from config, giving
a friendly error when the chosen backend's optional dependencies are missing.
"""

from __future__ import annotations

from distillation.config.models import StudentBackendKind, StudentConfig
from distillation.student.base import StudentBackend


def get_backend(config: StudentConfig) -> StudentBackend:
    """Instantiate the configured student backend.

    Args:
        config: The student section of an experiment config.

    Returns:
        A ready (but not yet loaded) :class:`StudentBackend`. Call
        :meth:`StudentBackend.load` and :meth:`apply_lora` before training.

    Raises:
        ImportError: If the backend's optional dependencies are not installed,
            with the exact ``pip install`` extra to fix it.
    """
    if config.backend is StudentBackendKind.MLX:
        try:
            from distillation.student.mlx_backend import MLXStudentBackend
        except ImportError as exc:  # pragma: no cover - env-dependent.
            raise ImportError(
                "MLX backend requires the 'mlx' extra: pip install -e '.[mlx]'"
            ) from exc
        return MLXStudentBackend(config.model)

    if config.backend is StudentBackendKind.PEFT:
        try:
            from distillation.student.peft_backend import PEFTStudentBackend
        except ImportError as exc:  # pragma: no cover - env-dependent.
            raise ImportError(
                "PEFT backend requires the 'peft' extra: pip install -e '.[peft]'"
            ) from exc
        return PEFTStudentBackend(config.model)

    raise ValueError(f"Unknown student backend: {config.backend!r}")  # pragma: no cover
