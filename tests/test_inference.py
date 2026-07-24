"""Tests for Phase 6 inference (engine + chat REPL) via FakeBackend."""

from __future__ import annotations

from pathlib import Path

import pytest
from distillation.config.models import GenerationConfig, StudentConfig
from distillation.inference.chat import run_chat
from distillation.inference.engine import InferenceEngine

from .conftest import FakeBackend


def _engine(backend: FakeBackend, adapter: Path | None = None) -> InferenceEngine:
    return InferenceEngine(backend, StudentConfig(), GenerationConfig(), adapter_path=adapter)


def test_engine_generates(fake_backend: FakeBackend) -> None:
    engine = _engine(fake_backend)
    result = engine.generate("hello")
    assert result.text == "student: hello"
    assert fake_backend.loaded


def test_engine_batch(fake_backend: FakeBackend) -> None:
    engine = _engine(fake_backend)
    results = engine.batch(["a", "b", "c"])
    assert [r.text for r in results] == ["student: a", "student: b", "student: c"]


def test_engine_loads_adapter_when_present(fake_backend: FakeBackend, tmp_path: Path) -> None:
    adapter = tmp_path / "adapter_final"
    engine = _engine(fake_backend, adapter=adapter)
    engine.generate("x")
    assert fake_backend.lora_applied  # adapter path -> LoRA re-attached


def test_engine_load_is_idempotent(fake_backend: FakeBackend) -> None:
    engine = _engine(fake_backend)
    engine.load()
    engine.load()  # must not reload / re-apply
    engine.generate("x")
    assert fake_backend.loaded


def test_chat_loop_exits_on_command(fake_backend: FakeBackend) -> None:
    engine = _engine(fake_backend)
    inputs = iter(["hello", "", "/exit"])
    outputs: list[str] = []
    run_chat(engine, input_fn=lambda _: next(inputs), output_fn=outputs.append)

    joined = "\n".join(outputs)
    assert "student: hello" in joined


def test_compare_once_collects_both_answers(fake_backend: FakeBackend) -> None:
    from distillation.inference.compare import compare_once

    from .conftest import FakeTeacher

    teacher = FakeTeacher()
    engine = _engine(fake_backend)
    c = compare_once(teacher, engine, "hello")

    assert c.teacher_text == "echo: hello"  # FakeTeacher echoes
    assert c.student_text == "student: hello"  # FakeBackend prefixes
    assert c.teacher_tokens == 3
    assert c.student_tokens == 4
    # speedup = teacher_latency / student_latency (0.01 / 0.02 with the fakes).
    assert c.speedup == pytest.approx(0.5)


def test_compare_loop_runs_until_exit(fake_backend: FakeBackend) -> None:
    from distillation.inference.compare import Comparison, run_compare_loop

    from .conftest import FakeTeacher

    teacher = FakeTeacher()
    engine = _engine(fake_backend)
    inputs = iter(["what is 2+2?", "/exit"])
    collected: list[Comparison] = []
    run_compare_loop(
        teacher,
        engine,
        render=collected.append,
        input_fn=lambda _: next(inputs),
        output_fn=lambda _: None,
    )
    assert len(collected) == 1
    assert collected[0].prompt == "what is 2+2?"


def test_chat_loop_handles_eof(fake_backend: FakeBackend) -> None:
    engine = _engine(fake_backend)

    def _raise_eof(_: str) -> str:
        raise EOFError

    outputs: list[str] = []
    run_chat(engine, input_fn=_raise_eof, output_fn=outputs.append)
    # Should exit cleanly without raising.
    assert outputs
