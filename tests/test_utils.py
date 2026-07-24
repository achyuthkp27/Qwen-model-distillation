"""Tests for utility helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from distillation.utils.io import read_jsonl, write_jsonl
from distillation.utils.memory import current_memory
from distillation.utils.timing import Stopwatch


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    rows = [{"id": 1, "x": "a"}, {"id": 2, "x": "b"}]
    assert write_jsonl(path, rows) == 2
    assert list(read_jsonl(path)) == rows


def test_jsonl_append(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [{"id": 1}])
    write_jsonl(path, [{"id": 2}], append=True)
    assert [r["id"] for r in read_jsonl(path)] == [1, 2]


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": 1}\n\n   \n{"id": 2}\n', encoding="utf-8")
    assert len(list(read_jsonl(path))) == 2


def test_jsonl_malformed_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": 1}\n{bad}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError, match="line 2"):
        list(read_jsonl(path))


def test_stopwatch_measures_positive_elapsed() -> None:
    with Stopwatch() as sw:
        sum(range(10_000))
    assert sw.elapsed_s >= 0.0
    assert sw.elapsed_ms == pytest.approx(sw.elapsed_s * 1000)


def test_stopwatch_requires_start() -> None:
    with pytest.raises(RuntimeError, match="before start"):
        Stopwatch().stop()


def test_current_memory_returns_positive() -> None:
    snap = current_memory()
    assert snap.rss_mb > 0
    assert snap.peak_rss_mb > 0
