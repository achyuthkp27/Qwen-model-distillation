"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from distillation.config.models import (
    GenerationConfig,
    LoRAConfig,
    TeacherConfig,
    TrainingConfig,
)
from distillation.datasets.schema import TokenUsage
from distillation.losses.base import DistillationStrategy
from distillation.student.base import (
    EncodedSample,
    GenerationResult,
    StepMetrics,
    StudentBackend,
)
from distillation.teacher.base import TeacherClient, TeacherResponse


class FakeTeacher(TeacherClient):
    """Deterministic in-memory teacher for tests (no network)."""

    def __init__(self, model: str = "fake:1b", *, fail_ids: set[str] | None = None) -> None:
        self._model = model
        self._fail_prompts = fail_ids or set()

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str, *, system: str | None = None) -> TeacherResponse:
        from distillation.teacher.base import TeacherError

        if prompt in self._fail_prompts:
            raise TeacherError(f"synthetic failure for {prompt!r}")
        return TeacherResponse(
            text=f"echo: {prompt}",
            usage=TokenUsage(prompt_tokens=len(prompt.split()), completion_tokens=3),
            latency_s=0.01,
        )

    def health_check(self) -> bool:
        return True


@pytest.fixture
def fake_teacher() -> FakeTeacher:
    """A working fake teacher."""
    return FakeTeacher()


@pytest.fixture
def teacher_config() -> TeacherConfig:
    """A minimal teacher config."""
    return TeacherConfig(model="qwen3:8b")


class FakeBackend(StudentBackend):
    """In-memory :class:`StudentBackend` for exercising the trainer on CPU.

    Simulates a monotonically decreasing loss and records every interface call so
    tests can assert the loop's control flow (accumulation, stepping, checkpoints)
    without MLX/PyTorch.
    """

    def __init__(self) -> None:
        self.loaded = False
        self.lora_applied = False
        self.optimizer_created = False
        self.accumulate_calls = 0
        self.apply_calls = 0
        self.zero_calls = 0
        self.eval_calls = 0
        self.saved_paths: list[Path] = []
        self._loss = 5.0

    def load(self) -> None:
        self.loaded = True

    def apply_lora(self, lora: LoRAConfig) -> None:
        self.lora_applied = True

    def encode(self, sample: dict[str, Any], max_seq_length: int) -> EncodedSample:
        return EncodedSample(input_ids=[1, 2, 3], labels=[2, 3, 4], loss_mask=[0, 1, 1])

    def collate(self, samples: list[EncodedSample]) -> Any:
        return {"n": len(samples), "tokens": sum(sum(s.loss_mask) for s in samples)}

    def create_optimizer(self, training: TrainingConfig) -> None:
        self.optimizer_created = True

    def accumulate_gradients(self, batch: Any, strategy: DistillationStrategy[Any]) -> StepMetrics:
        self.accumulate_calls += 1
        self._loss = max(0.0, self._loss - 0.01)
        return StepMetrics(loss=self._loss, num_tokens=batch["tokens"])

    def apply_gradients(self, learning_rate: float, max_grad_norm: float | None) -> float:
        self.apply_calls += 1
        return 1.23

    def zero_gradients(self) -> None:
        self.zero_calls += 1

    def eval_loss(self, batch: Any, strategy: DistillationStrategy[Any]) -> StepMetrics:
        self.eval_calls += 1
        return StepMetrics(loss=self._loss + 0.5, num_tokens=batch["tokens"])

    def generate(self, prompt: str, generation: GenerationConfig) -> GenerationResult:
        return GenerationResult(text=f"student: {prompt}", num_tokens=4, latency_s=0.02)

    def save_adapter(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-adapter", encoding="utf-8")
        self.saved_paths.append(path)

    def load_adapter(self, path: Path) -> None:
        pass

    def num_trainable_params(self) -> int:
        return 1000

    def memory_mb(self) -> float:
        return 42.0


@pytest.fixture
def fake_backend() -> FakeBackend:
    """A fake student backend for trainer tests."""
    return FakeBackend()
