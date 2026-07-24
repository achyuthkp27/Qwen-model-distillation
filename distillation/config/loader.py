"""YAML <-> :class:`ExperimentConfig` (de)serialization.

Kept separate from the models so the schema has no IO concerns and can be reused
(e.g. programmatically constructing configs in tests without touching disk).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from distillation.config.models import ExperimentConfig


def load_config(path: Path | str) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Args:
        path: Path to the ``.yaml`` config.

    Returns:
        A validated :class:`ExperimentConfig`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is empty, not a mapping, or fails validation.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")

    raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Config file is empty: {p}")
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}: {p}")

    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        # Surface a compact, human-readable summary alongside the full error.
        raise ValueError(f"Invalid configuration in {p}:\n{exc}") from exc


def dump_config(config: ExperimentConfig, path: Path | str) -> Path:
    """Serialize a config back to YAML (used for experiment provenance, Phase 7).

    Args:
        config: The config to persist.
        path: Destination ``.yaml`` file (parent dirs created as needed).

    Returns:
        The path written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Round-trip through JSON mode so Paths/enums become plain YAML scalars.
    data = json.loads(config.model_dump_json())
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p
