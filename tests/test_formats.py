"""Tests for Phase 2 format adapters."""

from __future__ import annotations

import pytest
from distillation.config.models import DatasetFormat
from distillation.datasets.formats import get_adapter
from distillation.datasets.schema import GenerationRecord


def _record(system: str | None = None) -> GenerationRecord:
    return GenerationRecord(
        id=1,
        prompt="Explain Redis.",
        system=system,
        response="Redis is an in-memory store.",
        teacher_model="qwen3:8b",
        latency_s=0.5,
    )


def test_instruction_adapter_shape() -> None:
    sample = get_adapter(DatasetFormat.INSTRUCTION).convert(_record())
    assert sample == {
        "instruction": "Explain Redis.",
        "input": "",
        "output": "Redis is an in-memory store.",
    }


def test_completion_adapter_shape() -> None:
    sample = get_adapter(DatasetFormat.COMPLETION).convert(_record())
    assert sample == {
        "prompt": "Explain Redis.",
        "completion": "Redis is an in-memory store.",
    }


def test_chat_adapter_without_system() -> None:
    sample = get_adapter(DatasetFormat.CHAT).convert(_record())
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["user", "assistant"]


def test_chat_adapter_record_system_wins_over_default() -> None:
    adapter = get_adapter(DatasetFormat.CHAT, system_prompt="default-sys")
    sample = adapter.convert(_record(system="record-sys"))
    assert sample["messages"][0] == {"role": "system", "content": "record-sys"}


def test_chat_adapter_falls_back_to_default_system() -> None:
    adapter = get_adapter(DatasetFormat.CHAT, system_prompt="default-sys")
    sample = adapter.convert(_record())
    assert sample["messages"][0] == {"role": "system", "content": "default-sys"}


def test_registry_covers_every_format() -> None:
    # Guards against adding a DatasetFormat without registering an adapter.
    for fmt in DatasetFormat:
        assert get_adapter(fmt).format is fmt


@pytest.mark.parametrize("fmt", list(DatasetFormat))
def test_all_adapters_json_serializable(fmt: DatasetFormat) -> None:
    import json

    sample = get_adapter(fmt).convert(_record(system="s"))
    json.dumps(sample)  # must not raise
