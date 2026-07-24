"""Tests for Phase 7 experiment management and Phase 3 schedule/metrics."""

from __future__ import annotations

from pathlib import Path

from distillation.config.models import ExperimentConfig
from distillation.experiment import ExperimentManager
from distillation.trainers.metrics import MetricRecord, MetricsRecorder
from distillation.trainers.schedule import warmup_constant
from distillation.utils.io import read_jsonl


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "exp_test",
            "output_root": str(tmp_path / "experiments"),
            "dataset": {"prompts_path": "prompts/seed_prompts.jsonl"},
        }
    )


def test_experiment_setup_creates_layout(tmp_path: Path) -> None:
    paths = ExperimentManager(_config(tmp_path)).setup()
    assert paths.logs.is_dir()
    assert paths.metrics.is_dir()
    assert paths.plots.is_dir()
    assert paths.adapters.is_dir()
    assert paths.config.is_file()  # config snapshot written


def test_experiment_config_snapshot_is_reloadable(tmp_path: Path) -> None:
    from distillation.config.loader import load_config

    paths = ExperimentManager(_config(tmp_path)).setup()
    reloaded = load_config(paths.config)
    assert reloaded.name == "exp_test"


def test_warmup_then_constant() -> None:
    schedule = warmup_constant(base_lr=1e-3, warmup_steps=10)
    assert schedule(5) == 5e-4  # halfway through warmup
    assert schedule(10) == 1e-3
    assert schedule(100) == 1e-3


def test_no_warmup_is_immediately_constant() -> None:
    schedule = warmup_constant(base_lr=2e-4, warmup_steps=0)
    assert schedule(1) == 2e-4


def test_metrics_recorder_persists_and_tracks(tmp_path: Path) -> None:
    recorder = MetricsRecorder(tmp_path / "metrics.jsonl")
    recorder.log(MetricRecord(step=1, epoch=0, split="train", loss=2.0))
    recorder.log(MetricRecord(step=1, epoch=0, split="val", loss=1.5))
    recorder.log(MetricRecord(step=2, epoch=0, split="val", loss=1.2))

    assert len(recorder.history) == 3
    assert recorder.best_val_loss() == 1.2
    assert len(recorder.by_split("val")) == 2
    assert len(list(read_jsonl(tmp_path / "metrics.jsonl"))) == 3


def test_metrics_recorder_truncates_prior_run(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"stale": true}\n', encoding="utf-8")
    MetricsRecorder(path)  # constructing should clear the file
    assert path.read_text(encoding="utf-8") == ""
