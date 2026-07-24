"""Experiment workspace management (Phase 7).

Each experiment owns a directory tree under ``output_root/<name>/`` holding its
config snapshot, logs, metrics, plots, and trained adapters. Centralizing the
layout here means every phase (train/eval/plot) agrees on where artifacts live.

    experiments/experiment_001/
    ├── config.yaml      # exact config used (provenance)
    ├── logs/            # rotating log files
    ├── metrics/         # metrics.jsonl, eval_results.json
    ├── plots/           # Phase 8 PNGs
    └── adapters/        # LoRA checkpoints
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from distillation.config.loader import dump_config
from distillation.config.models import ExperimentConfig
from distillation.utils.logging import get_logger

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    """Resolved artifact paths for one experiment."""

    root: Path
    config: Path
    logs: Path
    metrics: Path
    plots: Path
    adapters: Path

    @property
    def metrics_file(self) -> Path:
        """The training metrics JSONL."""
        return self.metrics / "metrics.jsonl"

    @property
    def eval_file(self) -> Path:
        """The evaluation results JSON."""
        return self.metrics / "eval_results.json"


class ExperimentManager:
    """Creates and exposes an experiment's directory layout (Phase 7).

    Args:
        config: The experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        root = config.experiment_dir
        self.paths = ExperimentPaths(
            root=root,
            config=root / "config.yaml",
            logs=root / "logs",
            metrics=root / "metrics",
            plots=root / "plots",
            adapters=root / "adapters",
        )

    def setup(self) -> ExperimentPaths:
        """Create all directories and snapshot the config for provenance.

        Returns:
            The :class:`ExperimentPaths` for this experiment.
        """
        for directory in (
            self.paths.root,
            self.paths.logs,
            self.paths.metrics,
            self.paths.plots,
            self.paths.adapters,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        dump_config(self._config, self.paths.config)
        _logger.info("Experiment '%s' ready at %s", self._config.name, self.paths.root)
        return self.paths
