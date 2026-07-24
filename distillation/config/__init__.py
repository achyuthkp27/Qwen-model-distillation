"""Typed, validated configuration for the whole framework.

Everything the framework does is driven by a single :class:`ExperimentConfig`
tree, loaded from YAML and validated by Pydantic. Sub-configs map 1:1 onto the
phases described in the project spec, so a config file reads like a plan.
"""

from __future__ import annotations

from distillation.config.loader import dump_config, load_config
from distillation.config.models import (
    DatasetConfig,
    DistillationConfig,
    EvaluationConfig,
    ExperimentConfig,
    GenerationConfig,
    LoRAConfig,
    OllamaConfig,
    StudentConfig,
    TeacherConfig,
    TrainingConfig,
)

__all__ = [
    "DatasetConfig",
    "DistillationConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "GenerationConfig",
    "LoRAConfig",
    "OllamaConfig",
    "StudentConfig",
    "TeacherConfig",
    "TrainingConfig",
    "dump_config",
    "load_config",
]
