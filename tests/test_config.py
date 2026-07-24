"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest
from distillation.config.loader import dump_config, load_config
from distillation.config.models import DatasetFormat, ExperimentConfig


def _minimal_yaml(tmp_path: Path) -> Path:
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        "name: test_exp\ndataset:\n  prompts_path: prompts/seed_prompts.jsonl\n",
        encoding="utf-8",
    )
    return cfg


def test_load_minimal_config_applies_defaults(tmp_path: Path) -> None:
    cfg = load_config(_minimal_yaml(tmp_path))
    assert cfg.name == "test_exp"
    assert cfg.teacher.model == "qwen3:8b"
    assert cfg.dataset.format is DatasetFormat.CHAT
    assert cfg.training.effective_batch_size == 8  # 1 * 8


def test_experiment_dir_derived_from_name(tmp_path: Path) -> None:
    cfg = load_config(_minimal_yaml(tmp_path))
    assert cfg.experiment_dir == Path("experiments") / "test_exp"


def test_unknown_field_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("name: x\nnope: 1\ndataset:\n  prompts_path: p.jsonl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_config(cfg)


def test_distillation_weight_validation() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        ExperimentConfig.model_validate(
            {
                "name": "x",
                "dataset": {"prompts_path": "p.jsonl"},
                "distillation": {"strategy": "logit", "alpha_ce": 0.7, "alpha_kd": 0.7},
            }
        )


def test_eval_prompts_path_defaults_none_and_accepts_override(tmp_path: Path) -> None:
    cfg = load_config(_minimal_yaml(tmp_path))
    assert cfg.evaluation.eval_prompts_path is None  # falls back to dataset prompts

    cfg2 = ExperimentConfig.model_validate(
        {
            "name": "x",
            "dataset": {"prompts_path": "p.jsonl"},
            "evaluation": {"eval_prompts_path": "held_out.jsonl"},
        }
    )
    assert cfg2.evaluation.eval_prompts_path == Path("held_out.jsonl")


def test_round_trip_dump_and_load(tmp_path: Path) -> None:
    cfg = load_config(_minimal_yaml(tmp_path))
    out = dump_config(cfg, tmp_path / "roundtrip.yaml")
    reloaded = load_config(out)
    assert reloaded.model_dump() == cfg.model_dump()
