"""Tests for the Phase 1 generation pipeline."""

from __future__ import annotations

from pathlib import Path

from distillation.datasets.generator import DatasetGenerator
from distillation.utils.io import read_jsonl, write_jsonl

from .conftest import FakeTeacher


def _write_prompts(path: Path, n: int) -> None:
    write_jsonl(path, [{"id": i, "prompt": f"prompt {i}"} for i in range(n)])


def test_generates_all_prompts(tmp_path: Path, fake_teacher: FakeTeacher) -> None:
    prompts = tmp_path / "prompts.jsonl"
    out = tmp_path / "gen.jsonl"
    _write_prompts(prompts, 5)

    summary = DatasetGenerator(fake_teacher).run(prompts, out)

    assert summary.total_prompts == 5
    assert summary.generated == 5
    assert summary.failed == 0
    rows = list(read_jsonl(out))
    assert len(rows) == 5
    assert rows[0]["response"] == "echo: prompt 0"
    assert rows[0]["teacher_model"] == "fake:1b"


def test_resume_skips_completed(tmp_path: Path, fake_teacher: FakeTeacher) -> None:
    prompts = tmp_path / "prompts.jsonl"
    out = tmp_path / "gen.jsonl"
    _write_prompts(prompts, 4)

    DatasetGenerator(fake_teacher).run(prompts, out, limit=2)
    summary = DatasetGenerator(fake_teacher).run(prompts, out)  # resume

    assert summary.skipped == 2
    assert summary.generated == 2
    assert len(list(read_jsonl(out))) == 4


def test_limit_is_target_total_not_per_run(tmp_path: Path, fake_teacher: FakeTeacher) -> None:
    # `limit` means "first N total": resuming with the same limit converges to N
    # and never exceeds it, regardless of prior partial runs.
    prompts = tmp_path / "prompts.jsonl"
    out = tmp_path / "gen.jsonl"
    _write_prompts(prompts, 10)

    # First run gets interrupted after 3 of the target 5.
    DatasetGenerator(fake_teacher).run(prompts, out, limit=3)
    # Resume with the same target total of 5.
    summary = DatasetGenerator(fake_teacher).run(prompts, out, limit=5)

    assert summary.skipped == 3  # the first 3 already done
    assert summary.generated == 2  # only the remaining 2 up to the target of 5
    ids = sorted(r["id"] for r in read_jsonl(out))
    assert ids == [0, 1, 2, 3, 4]  # exactly the first 5, never more

    # Re-running at the same limit is a no-op (idempotent).
    again = DatasetGenerator(fake_teacher).run(prompts, out, limit=5)
    assert again.generated == 0
    assert len(list(read_jsonl(out))) == 5


def test_concurrent_generation_completes_all(tmp_path: Path, fake_teacher: FakeTeacher) -> None:
    prompts = tmp_path / "prompts.jsonl"
    out = tmp_path / "gen.jsonl"
    _write_prompts(prompts, 20)

    summary = DatasetGenerator(fake_teacher).run(prompts, out, max_concurrency=4)

    assert summary.generated == 20
    assert summary.failed == 0
    # Every prompt id appears exactly once despite concurrent generation.
    ids = sorted(r["id"] for r in read_jsonl(out))
    assert ids == list(range(20))


def test_failures_recorded_not_raised(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    out = tmp_path / "gen.jsonl"
    _write_prompts(prompts, 3)
    teacher = FakeTeacher(fail_ids={"prompt 1"})

    summary = DatasetGenerator(teacher).run(prompts, out)

    assert summary.generated == 2
    assert summary.failed == 1
    failed = [r for r in read_jsonl(out) if r["error"] is not None]
    assert len(failed) == 1
    assert failed[0]["id"] == 1
