"""Evaluation metrics, comparison harness, reporting (Phase 5) and plots (8)."""

from __future__ import annotations

from distillation.evaluation.classification import (
    ClassificationReport,
    evaluate_accuracy,
    parse_label,
    score_predictions,
)
from distillation.evaluation.evaluator import (
    EvalResults,
    Evaluator,
    Generation,
    GenerationSource,
    SubjectResult,
)
from distillation.evaluation.metrics import TextMetric, get_text_metrics
from distillation.evaluation.plots import generate_plots
from distillation.evaluation.report import render_report
from distillation.evaluation.sources import EngineSource, StudentSource, TeacherSource

__all__ = [
    "ClassificationReport",
    "EngineSource",
    "EvalResults",
    "Evaluator",
    "Generation",
    "GenerationSource",
    "StudentSource",
    "SubjectResult",
    "TeacherSource",
    "TextMetric",
    "evaluate_accuracy",
    "generate_plots",
    "get_text_metrics",
    "parse_label",
    "render_report",
    "score_predictions",
]
