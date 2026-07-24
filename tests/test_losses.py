"""Tests for Phase 4 distillation strategies (backend-agnostic, NumPy math)."""

from __future__ import annotations

import numpy as np
import pytest
from distillation.config.models import DistillationConfig, DistillationStrategyKind
from distillation.losses.base import LossInputs, get_strategy
from distillation.losses.ops import NumpyOps
from distillation.losses.response import ResponseDistillation


def test_response_loss_matches_manual_cross_entropy() -> None:
    ops = NumpyOps()
    # 1 batch, 2 timesteps, vocab of 3.
    logits = np.array([[[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]])
    labels = np.array([[0, 1]])
    mask = np.array([[1.0, 1.0]])

    strat = ResponseDistillation(DistillationConfig())
    result = strat.compute_loss(
        LossInputs(student_logits=logits, labels=labels, loss_mask=mask), ops
    )

    # Manual reference: mean negative log-softmax at the target indices.
    logp = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    expected = -np.mean([logp[0, 0, 0], logp[0, 1, 1]])
    assert result.total == pytest.approx(expected)
    assert result.components["ce"] == pytest.approx(expected)


def test_response_loss_respects_mask() -> None:
    ops = NumpyOps()
    logits = np.random.default_rng(0).normal(size=(1, 4, 5))
    labels = np.array([[1, 2, 3, 4]])
    full_mask = np.array([[1.0, 1.0, 1.0, 1.0]])
    partial_mask = np.array([[0.0, 0.0, 1.0, 1.0]])

    strat = ResponseDistillation(DistillationConfig())
    full = strat.compute_loss(LossInputs(logits, labels, full_mask), ops).total
    partial = strat.compute_loss(LossInputs(logits, labels, partial_mask), ops).total
    # Masking changes which positions contribute -> different loss.
    assert not np.isclose(full, partial)


def test_get_strategy_returns_response() -> None:
    strat = get_strategy(DistillationConfig(strategy=DistillationStrategyKind.RESPONSE))
    assert isinstance(strat, ResponseDistillation)
    assert strat.kind is DistillationStrategyKind.RESPONSE


def test_unimplemented_strategy_raises_helpful_error() -> None:
    cfg = DistillationConfig(strategy=DistillationStrategyKind.LOGIT, alpha_ce=0.5, alpha_kd=0.5)
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        get_strategy(cfg)


def test_validate_inputs_flags_missing_signals() -> None:
    strat = ResponseDistillation(DistillationConfig())
    strat.requires_teacher_logits = True  # simulate a stricter strategy
    inputs = LossInputs(np.zeros((1, 1, 2)), np.zeros((1, 1), dtype=int), np.ones((1, 1)))
    with pytest.raises(ValueError, match="requires teacher_logits"):
        strat.validate_inputs(inputs)
