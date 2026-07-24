"""Phase 2 training-dataset builder.

Turns raw teacher generations into formatted, split train/val files:

1. Load :class:`GenerationRecord` rows and drop failed/empty ones.
2. Convert each via the selected :class:`FormatAdapter`.
3. Deterministically shuffle (seeded) and split into train/validation.
4. Stream the results to JSONL.

Everything is driven by :class:`DatasetConfig`, so `distill format` needs only
the experiment YAML.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from distillation.config.models import DatasetConfig
from distillation.datasets.formats import FormatAdapter, get_adapter
from distillation.datasets.schema import GenerationRecord
from distillation.utils.io import read_jsonl, write_jsonl
from distillation.utils.logging import get_logger

_logger = get_logger(__name__)


@dataclass(slots=True)
class BuildSummary:
    """Outcome of a dataset build.

    Attributes:
        total_records: Generation rows read from disk.
        skipped: Rows dropped because they failed/were empty.
        train_size: Samples written to the train file.
        val_size: Samples written to the validation file.
        train_path: Train output path.
        val_path: Validation output path (``None`` if no split requested).
    """

    total_records: int = 0
    skipped: int = 0
    train_size: int = 0
    val_size: int = 0
    train_path: Path | None = None
    val_path: Path | None = None


class DatasetBuilder:
    """Build formatted train/val datasets from teacher generations (Phase 2).

    Args:
        config: The dataset section of an experiment config.
        seed: Seed controlling the deterministic shuffle/split.
        adapter: Optional explicit adapter; by default one is built from
            ``config.format``. Injectable for testing / custom formats.
    """

    def __init__(
        self,
        config: DatasetConfig,
        *,
        seed: int = 42,
        adapter: FormatAdapter | None = None,
    ) -> None:
        self._config = config
        self._seed = seed
        self._adapter = adapter or get_adapter(config.format)

    def _load_valid_records(self, path: Path) -> tuple[list[GenerationRecord], int]:
        """Load generation records, returning (valid, skipped_count)."""
        valid: list[GenerationRecord] = []
        skipped = 0
        for row in read_jsonl(path):
            record = GenerationRecord.model_validate(row)
            if record.ok:
                valid.append(record)
            else:
                skipped += 1
        return valid, skipped

    def _split(
        self, records: list[GenerationRecord]
    ) -> tuple[list[GenerationRecord], list[GenerationRecord]]:
        """Shuffle (optionally) and split into train/val by ``val_fraction``."""
        items = list(records)
        if self._config.shuffle:
            # Local RNG so we never perturb global randomness elsewhere.
            random.Random(self._seed).shuffle(items)

        if self._config.val_path is None or self._config.val_fraction <= 0.0:
            return items, []

        n_val = int(len(items) * self._config.val_fraction)
        # Val is taken from the front post-shuffle; train is the remainder.
        return items[n_val:], items[:n_val]

    def build(self) -> BuildSummary:
        """Execute the full build and write train/val files.

        Returns:
            A :class:`BuildSummary` describing what was produced.
        """
        gen_path = self._config.generations_path
        records, skipped = self._load_valid_records(gen_path)
        _logger.info(
            "Loaded %d valid records (%d skipped) from %s; formatting as '%s'.",
            len(records),
            skipped,
            gen_path,
            self._config.format.value,
        )

        train_records, val_records = self._split(records)

        train_samples = (self._adapter.convert(r) for r in train_records)
        train_size = write_jsonl(self._config.train_path, train_samples)

        val_size = 0
        val_path: Path | None = None
        if val_records and self._config.val_path is not None:
            val_samples = (self._adapter.convert(r) for r in val_records)
            val_size = write_jsonl(self._config.val_path, val_samples)
            val_path = self._config.val_path

        _logger.info(
            "Wrote %d train -> %s | %d val -> %s",
            train_size,
            self._config.train_path,
            val_size,
            val_path,
        )
        return BuildSummary(
            total_records=len(records) + skipped,
            skipped=skipped,
            train_size=train_size,
            val_size=val_size,
            train_path=self._config.train_path,
            val_path=val_path,
        )
