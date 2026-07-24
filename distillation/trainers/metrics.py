"""Training-metrics recording (Phases 3, 7, 8).

Every logged step is appended to a JSONL file *and* retained in memory. The
JSONL is the durable record (survives crashes, feeds the HTML report); the
in-memory history feeds the Phase 8 plots. One recorder, both consumers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from distillation.utils.io import write_jsonl
from distillation.utils.logging import get_logger

_logger = get_logger(__name__)


@dataclass(slots=True)
class MetricRecord:
    """A single logged training/validation point.

    Attributes:
        step: Global optimizer step.
        epoch: Current epoch (0-indexed).
        split: ``"train"`` or ``"val"``.
        loss: Scalar loss at this point.
        learning_rate: LR used for this step (train only; 0 for val).
        grad_norm: Pre-clip gradient norm (train only; 0 for val).
        tokens_per_sec: Training throughput (train only).
        memory_mb: Model memory footprint in MiB.
    """

    step: int
    epoch: int
    split: str
    loss: float
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_per_sec: float = 0.0
    memory_mb: float = 0.0


class MetricsRecorder:
    """Append-only metrics sink with in-memory history.

    Args:
        path: JSONL file the records are appended to (parent dirs created).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._history: list[MetricRecord] = []
        # Truncate any prior run's file so a fresh training starts clean.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def log(self, record: MetricRecord) -> None:
        """Persist and retain one metric record."""
        self._history.append(record)
        write_jsonl(self._path, [asdict(record)], append=True)

    @property
    def path(self) -> Path:
        """The JSONL file records are written to."""
        return self._path

    @property
    def history(self) -> list[MetricRecord]:
        """All records logged so far, in order."""
        return self._history

    def by_split(self, split: str) -> list[MetricRecord]:
        """Records for a given split (``"train"`` / ``"val"``)."""
        return [r for r in self._history if r.split == split]

    def best_val_loss(self) -> float | None:
        """Lowest validation loss recorded, or ``None`` if no val points."""
        val = [r.loss for r in self._history if r.split == "val"]
        return min(val) if val else None
