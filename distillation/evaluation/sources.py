"""Adapters exposing teacher/student as :class:`GenerationSource` (Phase 5).

These thin wrappers let the evaluator treat an Ollama teacher and an MLX/PEFT
student uniformly, without knowing either concrete type.
"""

from __future__ import annotations

from distillation.config.models import GenerationConfig
from distillation.evaluation.evaluator import Generation
from distillation.inference.engine import InferenceEngine
from distillation.student.base import StudentBackend
from distillation.teacher.base import TeacherClient


class TeacherSource:
    """Expose a :class:`TeacherClient` as a generation source."""

    def __init__(self, client: TeacherClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        """Subject name."""
        return "teacher"

    def generate(self, prompt: str) -> Generation:
        """Generate one completion via the teacher."""
        resp = self._client.generate(prompt)
        return Generation(
            text=resp.text,
            num_tokens=resp.usage.completion_tokens,
            latency_s=resp.latency_s,
        )


class StudentSource:
    """Expose a :class:`StudentBackend` as a generation source.

    Args:
        backend: The (loaded) student backend.
        generation: Sampling parameters for generation.
        name: Subject label (e.g. ``student_before`` / ``student_after``).
    """

    def __init__(self, backend: StudentBackend, generation: GenerationConfig, name: str) -> None:
        self._backend = backend
        self._generation = generation
        self._name = name

    @property
    def name(self) -> str:
        """Subject name."""
        return self._name

    def generate(self, prompt: str) -> Generation:
        """Generate one completion via the student."""
        result = self._backend.generate(prompt, self._generation)
        return Generation(
            text=result.text,
            num_tokens=result.num_tokens,
            latency_s=result.latency_s,
            memory_mb=self._backend.memory_mb(),
        )


class EngineSource:
    """Expose an :class:`InferenceEngine` (student + loaded adapter) as a source.

    Unlike :class:`StudentSource` (a raw backend), this loads the trained adapter
    automatically, so it evaluates the *distilled* student when one exists.

    Args:
        engine: The inference engine (loads its adapter on first generate).
        name: Subject label.
    """

    def __init__(self, engine: InferenceEngine, name: str = "student") -> None:
        self._engine = engine
        self._name = name

    @property
    def name(self) -> str:
        """Subject name."""
        return self._name

    def generate(self, prompt: str) -> Generation:
        """Generate one completion via the student engine."""
        result = self._engine.generate(prompt)
        return Generation(
            text=result.text, num_tokens=result.num_tokens, latency_s=result.latency_s
        )
