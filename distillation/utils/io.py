"""JSONL read/write helpers.

JSONL is the lingua franca of this framework: prompts in, teacher generations
out, training samples, and evaluation records all flow as line-delimited JSON.
These helpers stream rather than loading whole files into memory so large
datasets remain tractable on a 24 GB machine.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Lazily yield objects from a JSONL file, skipping blank lines.

    Args:
        path: Source ``.jsonl`` file.

    Yields:
        One decoded JSON object per non-empty line.

    Raises:
        FileNotFoundError: If the path does not exist.
        json.JSONDecodeError: If a line contains malformed JSON (the line
            number is included in the raised error's message).
    """
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:  # Re-raise with context.
                raise json.JSONDecodeError(
                    f"{exc.msg} (in {p} at line {lineno})", exc.doc, exc.pos
                ) from exc


def write_jsonl(
    path: Path | str,
    rows: Iterable[dict[str, Any]],
    *,
    append: bool = False,
) -> int:
    """Write an iterable of JSON-serializable objects as JSONL.

    Parent directories are created automatically.

    Args:
        path: Destination ``.jsonl`` file.
        rows: Iterable of JSON-serializable dictionaries.
        append: If ``True``, append instead of truncating. Enables incremental,
            crash-resilient dataset generation.

    Returns:
        The number of rows written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with p.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count
