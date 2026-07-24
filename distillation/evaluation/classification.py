"""Classification accuracy evaluation (for labelled tasks like sentiment).

Unlike open-ended generation (where we can only measure *similarity* to the
teacher), a classification task has a single **gold label** per example — so we
can compute a real **accuracy**. This module:

1. Parses a free-text model answer into one of the allowed labels
   (models wrap the label in punctuation, reasoning tags, or a sentence).
2. Scores predictions against the gold labels, with a per-class breakdown and a
   confusion matrix.

It reuses the :class:`GenerationSource` protocol, so the same teacher/student
wrappers used elsewhere plug straight in.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from distillation.evaluation.evaluator import GenerationSource
from distillation.utils.logging import get_logger

_logger = get_logger(__name__)

#: The allowed sentiment labels (order = canonical order for reports).
SENTIMENT_LABELS = ("positive", "negative", "neutral")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_label(text: str, labels: tuple[str, ...] = SENTIMENT_LABELS) -> str | None:
    """Extract the predicted label from a model's free-text answer.

    Strips any reasoning ``<think>…</think>`` block, lowercases, and returns the
    first allowed label word that appears. Returns ``None`` if the answer
    contains no recognizable label (counted as wrong, but tracked separately).

    Args:
        text: The raw model output.
        labels: Allowed labels to look for.

    Returns:
        The matched label, or ``None`` if unparseable.
    """
    cleaned = _THINK_RE.sub(" ", text or "").lower()
    best: tuple[int, str] | None = None
    for label in labels:
        # Word-boundary match so "neutral" doesn't match inside another word.
        m = re.search(rf"\b{re.escape(label)}\b", cleaned)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), label)
    return best[1] if best else None


@dataclass(slots=True)
class ClassificationReport:
    """Accuracy results for one subject on a labelled test set.

    Attributes:
        name: Subject name (e.g. 'teacher').
        total: Number of test examples.
        correct: Predictions matching the gold label.
        unparsed: Answers with no recognizable label (counted as wrong).
        per_class: {label: (correct, total)} — recall per class.
        confusion: {(gold, predicted-or-'?'): count}.
        predictions: The parsed prediction per example (for inspection).
    """

    name: str
    total: int = 0
    correct: int = 0
    unparsed: int = 0
    per_class: dict[str, tuple[int, int]] = field(default_factory=dict)
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    predictions: list[str | None] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Fraction correct over all examples (unparsed count as wrong)."""
        return self.correct / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, object]:
        """JSON-serializable summary."""
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "correct": self.correct,
            "total": self.total,
            "unparsed": self.unparsed,
            "per_class_recall": {k: (c / t if t else 0.0) for k, (c, t) in self.per_class.items()},
        }


def score_predictions(
    name: str,
    predictions: list[str | None],
    golds: list[str],
    labels: tuple[str, ...] = SENTIMENT_LABELS,
) -> ClassificationReport:
    """Score already-parsed predictions against gold labels.

    Split out from generation so it is trivially unit-testable.
    """
    report = ClassificationReport(name=name, total=len(golds), predictions=predictions)
    per_correct: Counter[str] = Counter()
    per_total: Counter[str] = Counter()
    for pred, gold in zip(predictions, golds, strict=True):
        per_total[gold] += 1
        report.confusion[(gold, pred or "?")] = report.confusion.get((gold, pred or "?"), 0) + 1
        if pred is None:
            report.unparsed += 1
        elif pred == gold:
            report.correct += 1
            per_correct[gold] += 1
    report.per_class = {lbl: (per_correct[lbl], per_total[lbl]) for lbl in labels if per_total[lbl]}
    return report


def evaluate_accuracy(
    source: GenerationSource,
    prompts: list[str],
    golds: list[str],
    labels: tuple[str, ...] = SENTIMENT_LABELS,
) -> ClassificationReport:
    """Run a subject over the prompts and score its accuracy against gold labels.

    Args:
        source: Teacher or student generation source.
        prompts: The classification prompts.
        golds: The gold label per prompt (same order).
        labels: Allowed labels.

    Returns:
        A :class:`ClassificationReport`.
    """
    predictions = [parse_label(source.generate(p).text, labels) for p in prompts]
    report = score_predictions(source.name, predictions, golds, labels)
    _logger.info(
        "%s accuracy: %.1f%% (%d/%d, %d unparsed)",
        source.name,
        report.accuracy * 100,
        report.correct,
        report.total,
        report.unparsed,
    )
    return report
