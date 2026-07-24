"""Comparative evaluation harness (Phase 5).

Runs the same prompt set through up to three subjects — **teacher**, **student
before training**, and **student after training** — and reports, per subject:

* text-similarity metrics vs. the teacher (BLEU / ROUGE / BERTScore),
* performance metrics (mean latency, tokens/sec, peak memory).

Subjects are supplied as :class:`GenerationSource` objects, so the harness is
decoupled from Ollama/MLX/PEFT specifics and is easily faked in tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from distillation.evaluation.metrics import TextMetric
from distillation.utils.logging import get_logger

_logger = get_logger(__name__)


@dataclass(slots=True)
class Generation:
    """One generation with its performance measurements."""

    text: str
    num_tokens: int
    latency_s: float
    memory_mb: float = 0.0

    @property
    def tokens_per_sec(self) -> float:
        """Tokens per second for this generation."""
        if self.latency_s <= 0 or self.num_tokens <= 0:
            return 0.0
        return self.num_tokens / self.latency_s


class GenerationSource(Protocol):
    """Anything that can produce a :class:`Generation` for a prompt."""

    @property
    def name(self) -> str:
        """Human-readable subject name (e.g. 'teacher')."""
        ...

    def generate(self, prompt: str) -> Generation:
        """Generate for a single prompt."""
        ...


@dataclass(slots=True)
class SubjectResult:
    """Aggregate evaluation result for one subject.

    Attributes:
        name: Subject name.
        responses: Generated text per prompt.
        mean_latency_s: Mean per-prompt latency.
        mean_tokens_per_sec: Mean throughput.
        peak_memory_mb: Peak memory observed across prompts.
        text_metrics: Text-similarity scores vs the teacher (empty for teacher).
    """

    name: str
    responses: list[str]
    mean_latency_s: float
    mean_tokens_per_sec: float
    peak_memory_mb: float
    text_metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "name": self.name,
            "mean_latency_s": self.mean_latency_s,
            "mean_tokens_per_sec": self.mean_tokens_per_sec,
            "peak_memory_mb": self.peak_memory_mb,
            "text_metrics": self.text_metrics,
            "num_responses": len(self.responses),
        }


@dataclass(slots=True)
class EvalResults:
    """Full comparison across all evaluated subjects."""

    prompts: list[str]
    subjects: list[SubjectResult]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "num_prompts": len(self.prompts),
            "subjects": [s.as_dict() for s in self.subjects],
        }


class Evaluator:
    """Generates for each subject and computes comparison metrics.

    Args:
        text_metrics: Metrics comparing student output to the teacher reference.
    """

    def __init__(self, text_metrics: list[TextMetric]) -> None:
        self._text_metrics = text_metrics

    def _run_subject(
        self, source: GenerationSource, prompts: list[str]
    ) -> tuple[list[Generation], SubjectResult]:
        """Generate for every prompt and aggregate performance metrics."""
        gens = [source.generate(p) for p in prompts]
        n = max(len(gens), 1)
        result = SubjectResult(
            name=source.name,
            responses=[g.text for g in gens],
            mean_latency_s=sum(g.latency_s for g in gens) / n,
            mean_tokens_per_sec=sum(g.tokens_per_sec for g in gens) / n,
            peak_memory_mb=max((g.memory_mb for g in gens), default=0.0),
        )
        return gens, result

    def _score_text(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Run every text metric, merging their score dicts."""
        scores: dict[str, float] = {}
        for metric in self._text_metrics:
            try:
                scores.update(metric.compute(predictions, references))
            except ImportError as exc:
                _logger.warning("Skipping metric '%s': %s", metric.name, exc)
        return scores

    def run(
        self,
        prompts: list[str],
        teacher: GenerationSource,
        students: Sequence[GenerationSource],
    ) -> EvalResults:
        """Evaluate the teacher and each student against the teacher reference.

        Args:
            prompts: Evaluation prompts.
            teacher: The teacher generation source (defines the reference text).
            students: Student subjects (e.g. before/after training).

        Returns:
            An :class:`EvalResults` with one :class:`SubjectResult` per subject.
        """
        _logger.info("Evaluating %d prompts across %d subjects", len(prompts), 1 + len(students))
        _, teacher_result = self._run_subject(teacher, prompts)
        references = teacher_result.responses

        subjects = [teacher_result]
        for student in students:
            _, result = self._run_subject(student, prompts)
            result.text_metrics = self._score_text(result.responses, references)
            subjects.append(result)

        return EvalResults(prompts=prompts, subjects=subjects)
