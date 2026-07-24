"""Tests for the Phase 3 training loop (backend-agnostic, via FakeBackend)."""

from __future__ import annotations

from pathlib import Path

from distillation.config.models import DistillationConfig, TrainingConfig
from distillation.losses.base import get_strategy
from distillation.trainers.metrics import MetricsRecorder
from distillation.trainers.trainer import DistillationTrainer

from .conftest import FakeBackend


def _make_trainer(
    backend: FakeBackend, tmp_path: Path, **cfg_overrides: object
) -> DistillationTrainer:
    training = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        epochs=1,
        log_interval=1,
        eval_interval=1,
        checkpoint_interval=2,
        **cfg_overrides,
    )
    recorder = MetricsRecorder(tmp_path / "metrics.jsonl")
    strategy = get_strategy(DistillationConfig())
    return DistillationTrainer(
        backend, strategy, training, recorder=recorder, adapter_dir=tmp_path / "adapters"
    )


def _samples(n: int) -> list[dict[str, object]]:
    return [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(n)]


def test_optimizer_steps_respect_accumulation(fake_backend: FakeBackend, tmp_path: Path) -> None:
    trainer = _make_trainer(fake_backend, tmp_path)
    # 8 samples, batch_size 1 -> 8 micro-batches; accum 2 -> 4 optimizer steps.
    result = trainer.train(_samples(8))

    assert fake_backend.accumulate_calls == 8
    assert fake_backend.apply_calls == 4
    assert fake_backend.zero_calls == 4
    assert result.total_steps == 4


def test_checkpoints_written(fake_backend: FakeBackend, tmp_path: Path) -> None:
    trainer = _make_trainer(fake_backend, tmp_path)
    result = trainer.train(_samples(8))

    # checkpoint_interval=2 over 4 steps -> steps 2 and 4, plus a final adapter.
    # The trainer passes suffix-less base paths; each backend adds its own convention.
    assert (tmp_path / "adapters" / "adapter_2").exists()
    assert (tmp_path / "adapters" / "adapter_4").exists()
    assert result.adapter_path.name == "adapter_final"
    assert result.adapter_path.exists()


def test_validation_logged(fake_backend: FakeBackend, tmp_path: Path) -> None:
    trainer = _make_trainer(fake_backend, tmp_path)
    result = trainer.train(_samples(4), val_samples=_samples(2))

    assert fake_backend.eval_calls > 0
    assert result.best_val_loss is not None


def test_metrics_persisted_to_disk(fake_backend: FakeBackend, tmp_path: Path) -> None:
    from distillation.utils.io import read_jsonl

    trainer = _make_trainer(fake_backend, tmp_path)
    trainer.train(_samples(4))

    rows = list(read_jsonl(tmp_path / "metrics.jsonl"))
    train_rows = [r for r in rows if r["split"] == "train"]
    assert train_rows
    assert train_rows[0]["grad_norm"] == 1.23
    assert train_rows[0]["memory_mb"] == 42.0


def test_loss_decreases_over_training(fake_backend: FakeBackend, tmp_path: Path) -> None:
    trainer = _make_trainer(fake_backend, tmp_path)
    result = trainer.train(_samples(20))
    assert result.final_train_loss < 5.0
