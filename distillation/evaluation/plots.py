"""Training visualizations (Phase 8).

Reads a metrics JSONL (produced by :class:`MetricsRecorder`) and renders the
standard training plots — loss curve, learning rate, validation loss, memory
usage, and throughput — as PNGs. ``matplotlib`` is imported lazily so the
framework loads without the ``eval`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from distillation.utils.io import read_jsonl
from distillation.utils.logging import get_logger

_logger = get_logger(__name__)


def _series(rows: list[dict[str, Any]], split: str, y_key: str) -> tuple[list[float], list[float]]:
    """Extract (step, y) pairs for a split where ``y_key`` is meaningful."""
    xs, ys = [], []
    for r in rows:
        if r.get("split") != split:
            continue
        xs.append(float(r["step"]))
        ys.append(float(r.get(y_key, 0.0)))
    return xs, ys


def generate_plots(metrics_path: Path, output_dir: Path) -> list[Path]:
    """Render all training plots from a metrics JSONL.

    Args:
        metrics_path: The ``metrics.jsonl`` file.
        output_dir: Directory to write PNGs into.

    Returns:
        Paths of the plots that were successfully written (empty if no data or
        matplotlib is unavailable).
    """
    if not metrics_path.is_file():
        _logger.warning("No metrics file at %s; skipping plots.", metrics_path)
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")  # Headless backend; no display needed.
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - env-dependent.
        _logger.warning("matplotlib not installed; skipping plots (pip install '.[eval]').")
        return []

    rows = list(read_jsonl(metrics_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # (filename, title, y-axis label, train y-key, include-val overlay)
    specs = [
        ("loss_curve", "Training loss", "loss", "loss", True),
        ("learning_rate", "Learning rate", "lr", "learning_rate", False),
        ("memory_usage", "Memory usage", "MiB", "memory_mb", False),
        ("throughput", "Training throughput", "tokens/sec", "tokens_per_sec", False),
    ]

    for filename, title, ylabel, y_key, overlay_val in specs:
        xs, ys = _series(rows, "train", y_key)
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, label="train", color="#1f77b4")
        if overlay_val:
            vx, vy = _series(rows, "val", y_key)
            if vx:
                ax.plot(vx, vy, label="val", color="#d62728", marker="o", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = output_dir / f"{filename}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(out)

    _logger.info("Wrote %d plots -> %s", len(written), output_dir)
    return written
