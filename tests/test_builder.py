"""Tests for the Phase 2 dataset builder."""

from __future__ import annotations

from pathlib import Path

from distillation.config.models import DatasetConfig, DatasetFormat
from distillation.datasets.builder import DatasetBuilder
from distillation.datasets.schema import GenerationRecord
from distillation.utils.io import read_jsonl, write_jsonl


def _write_generations(path: Path, n: int, *, n_failed: int = 0) -> None:
    rows = []
    for i in range(n):
        rows.append(
            GenerationRecord(
                id=i,
                prompt=f"q{i}",
                response=f"a{i}",
                teacher_model="qwen3:8b",
                latency_s=0.1,
            ).model_dump()
        )
    for j in range(n_failed):
        rows.append(
            GenerationRecord(
                id=1000 + j,
                prompt=f"bad{j}",
                response="",
                teacher_model="qwen3:8b",
                latency_s=0.0,
                error="boom",
            ).model_dump()
        )
    write_jsonl(path, rows)


def _config(tmp_path: Path, **overrides: object) -> DatasetConfig:
    base: dict[str, object] = {
        "prompts_path": tmp_path / "prompts.jsonl",
        "generations_path": tmp_path / "gen.jsonl",
        "train_path": tmp_path / "train.jsonl",
        "val_path": tmp_path / "val.jsonl",
        "format": DatasetFormat.CHAT,
        "val_fraction": 0.2,
        "shuffle": True,
    }
    base.update(overrides)
    return DatasetConfig.model_validate(base)


def test_build_splits_train_and_val(tmp_path: Path) -> None:
    _write_generations(tmp_path / "gen.jsonl", 10)
    summary = DatasetBuilder(_config(tmp_path), seed=42).build()

    assert summary.train_size == 8
    assert summary.val_size == 2
    assert len(list(read_jsonl(tmp_path / "train.jsonl"))) == 8
    assert len(list(read_jsonl(tmp_path / "val.jsonl"))) == 2


def test_build_skips_failed_records(tmp_path: Path) -> None:
    _write_generations(tmp_path / "gen.jsonl", 5, n_failed=3)
    summary = DatasetBuilder(_config(tmp_path, val_fraction=0.0, val_path=None)).build()

    assert summary.skipped == 3
    assert summary.train_size == 5
    assert summary.val_size == 0


def test_build_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    _write_generations(tmp_path / "gen.jsonl", 20)
    cfg = _config(tmp_path)

    DatasetBuilder(cfg, seed=7).build()
    first = list(read_jsonl(tmp_path / "train.jsonl"))
    DatasetBuilder(cfg, seed=7).build()
    second = list(read_jsonl(tmp_path / "train.jsonl"))

    assert first == second


def test_no_val_path_produces_only_train(tmp_path: Path) -> None:
    _write_generations(tmp_path / "gen.jsonl", 6)
    summary = DatasetBuilder(_config(tmp_path, val_path=None)).build()

    assert summary.val_size == 0
    assert summary.train_size == 6
    assert not (tmp_path / "val.jsonl").exists()


def test_completion_format_output(tmp_path: Path) -> None:
    _write_generations(tmp_path / "gen.jsonl", 4)
    DatasetBuilder(_config(tmp_path, format=DatasetFormat.COMPLETION, val_path=None)).build()

    rows = list(read_jsonl(tmp_path / "train.jsonl"))
    assert set(rows[0].keys()) == {"prompt", "completion"}
