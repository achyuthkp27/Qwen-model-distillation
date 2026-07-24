"""Inference engine (Phase 6).

Wraps a student :class:`StudentBackend` for serving: loads the base model,
re-attaches the LoRA structure, optionally loads a trained adapter, and exposes
single / batch generation. Shared by the CLI, REST API, and chat REPL so they
never duplicate load logic.
"""

from __future__ import annotations

from pathlib import Path

from distillation.config.models import GenerationConfig, StudentConfig
from distillation.student.base import GenerationResult, StudentBackend
from distillation.student.factory import get_backend
from distillation.utils.logging import get_logger

_logger = get_logger(__name__)


class InferenceEngine:
    """Load a (optionally adapted) student and generate completions.

    Args:
        backend: A student backend (not yet loaded).
        student: Student config (provides the LoRA structure to re-attach).
        generation: Default sampling parameters.
        adapter_path: Optional trained adapter to load after attaching LoRA.
    """

    def __init__(
        self,
        backend: StudentBackend,
        student: StudentConfig,
        generation: GenerationConfig,
        *,
        adapter_path: Path | None = None,
    ) -> None:
        self._backend = backend
        self._student = student
        self._generation = generation
        self._adapter_path = adapter_path
        self._loaded = False

    @classmethod
    def from_config(
        cls,
        student: StudentConfig,
        generation: GenerationConfig,
        *,
        adapter_path: Path | None = None,
    ) -> InferenceEngine:
        """Build an engine from config, selecting the backend automatically."""
        return cls(get_backend(student), student, generation, adapter_path=adapter_path)

    def load(self) -> None:
        """Load the model and adapter (idempotent)."""
        if self._loaded:
            return
        self._backend.load()
        if self._adapter_path is not None:
            # Re-attach the LoRA structure, then load the trained weights into it.
            self._backend.apply_lora(self._student.lora)
            self._backend.load_adapter(self._adapter_path)
            _logger.info("Loaded adapter from %s", self._adapter_path)
        self._loaded = True

    def generate(self, prompt: str) -> GenerationResult:
        """Generate a completion for one prompt (loads on first use)."""
        self.load()
        return self._backend.generate(prompt, self._generation)

    def batch(self, prompts: list[str]) -> list[GenerationResult]:
        """Generate completions for a list of prompts sequentially."""
        self.load()
        return [self._backend.generate(p, self._generation) for p in prompts]

    def close(self) -> None:
        """Release backend resources."""
        self._backend.close()
