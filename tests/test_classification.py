"""Tests for classification accuracy evaluation (label parsing + scoring)."""

from __future__ import annotations

from distillation.evaluation.classification import (
    evaluate_accuracy,
    parse_label,
    score_predictions,
)
from distillation.evaluation.evaluator import Generation


def test_parse_label_plain() -> None:
    assert parse_label("positive") == "positive"
    assert parse_label("Negative.") == "negative"
    assert parse_label("The sentiment is Neutral") == "neutral"


def test_parse_label_strips_reasoning_tags() -> None:
    assert parse_label("<think>hmm the user is happy</think>\npositive") == "positive"


def test_parse_label_word_boundary() -> None:
    # 'neutrality' should not match 'neutral' as the intended answer word...
    assert parse_label("positivity vibes → positive") == "positive"
    # ...but a real label word is found.
    assert parse_label("no clear label here") is None


def test_parse_label_first_occurrence_wins() -> None:
    assert parse_label("negative, definitely not positive") == "negative"


def test_score_predictions_accuracy_and_breakdown() -> None:
    preds = ["positive", "negative", "neutral", "positive", None]
    golds = ["positive", "negative", "neutral", "negative", "neutral"]
    r = score_predictions("student", preds, golds)

    assert r.total == 5
    assert r.correct == 3  # first three right
    assert r.unparsed == 1  # the None
    assert r.accuracy == 0.6
    assert r.per_class["positive"] == (1, 1)
    assert r.per_class["negative"] == (1, 2)


def test_evaluate_accuracy_end_to_end() -> None:
    # Fake source that "answers" by echoing a canned label per prompt.
    answers = {"p1": "positive", "p2": "the sentiment is negative", "p3": "??"}

    class _Src:
        name = "teacher"

        def generate(self, prompt: str) -> Generation:
            return Generation(text=answers[prompt], num_tokens=1, latency_s=0.0)

    report = evaluate_accuracy(_Src(), ["p1", "p2", "p3"], ["positive", "negative", "neutral"])
    assert report.correct == 2
    assert report.unparsed == 1
    assert report.accuracy == 2 / 3
