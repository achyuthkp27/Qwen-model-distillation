"""Tests for Phase 5 evaluation + Phase 8 plotting (no heavy deps)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from distillation.evaluation.evaluator import EvalResults, Evaluator, Generation
from distillation.evaluation.metrics import TextMetric, get_text_metrics
from distillation.evaluation.plots import generate_plots
from distillation.evaluation.report import render_report
from distillation.utils.io import write_jsonl


class _FakeSource:
    """A canned generation source for tests."""

    def __init__(self, name: str, texts: list[str], latency: float = 0.1) -> None:
        self._name = name
        self._texts = texts
        self._latency = latency
        self._i = 0

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str) -> Generation:
        text = self._texts[self._i % len(self._texts)]
        self._i += 1
        return Generation(text=text, num_tokens=len(text.split()), latency_s=self._latency)


class _ExactMatch(TextMetric):
    """Dependency-free metric: fraction of exact matches."""

    @property
    def name(self) -> str:
        return "exact_match"

    def compute(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        hits = sum(p == r for p, r in zip(predictions, references, strict=True))
        return {"exact_match": hits / max(len(predictions), 1)}


def test_evaluator_scores_students_against_teacher() -> None:
    prompts = ["a", "b"]
    teacher = _FakeSource("teacher", ["ANS-a", "ANS-b"], latency=0.2)
    before = _FakeSource("student_before", ["wrong", "wrong"], latency=0.05)
    after = _FakeSource("student_after", ["ANS-a", "ANS-b"], latency=0.05)

    results = Evaluator([_ExactMatch()]).run(prompts, teacher, [before, after])

    assert [s.name for s in results.subjects] == ["teacher", "student_before", "student_after"]
    # Teacher has no self-metrics; after-training matches the teacher exactly.
    assert results.subjects[0].text_metrics == {}
    assert results.subjects[1].text_metrics["exact_match"] == 0.0
    assert results.subjects[2].text_metrics["exact_match"] == 1.0


def test_evaluator_measures_performance() -> None:
    teacher = _FakeSource("teacher", ["hello world"], latency=0.5)
    results = Evaluator([]).run(["p"], teacher, [])
    subj = results.subjects[0]
    assert subj.mean_latency_s == pytest.approx(0.5)
    assert subj.mean_tokens_per_sec == pytest.approx(2 / 0.5)


def test_get_text_metrics_filters_performance_names() -> None:
    metrics = get_text_metrics(["latency", "tokens_per_sec", "memory"])
    assert metrics == []


def test_get_text_metrics_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown metric"):
        get_text_metrics(["nonsense"])


def test_render_report_writes_html(tmp_path: Path) -> None:
    teacher = _FakeSource("teacher", ["x"], latency=0.1)
    after = _FakeSource("student_after", ["x"], latency=0.05)
    results = Evaluator([_ExactMatch()]).run(["p"], teacher, [after])

    out = render_report(results, tmp_path / "report.html", experiment_name="exp_test")
    content = out.read_text(encoding="utf-8")
    assert out.exists()
    assert "exp_test" in content
    assert "student_after" in content
    assert "Tokens/sec" in content


def test_results_serialize_to_json() -> None:
    teacher = _FakeSource("teacher", ["x"])
    results = Evaluator([]).run(["p"], teacher, [])
    payload = json.dumps(results.as_dict())
    assert "subjects" in payload


def test_generate_plots_no_matplotlib_or_data_is_safe(tmp_path: Path) -> None:
    # Empty metrics file -> no crash, returns whatever plots are possible.
    metrics = tmp_path / "metrics.jsonl"
    write_jsonl(metrics, [{"step": 1, "split": "train", "loss": 2.0, "learning_rate": 1e-4}])
    written = generate_plots(metrics, tmp_path / "plots")
    # Either matplotlib is installed (>=1 plot) or not (0); both are valid.
    assert isinstance(written, list)


def test_empty_results_object() -> None:
    empty = EvalResults(prompts=[], subjects=[])
    assert empty.as_dict()["num_prompts"] == 0
