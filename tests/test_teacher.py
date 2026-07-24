"""Tests for the Ollama teacher client and reasoning-model handling."""

from __future__ import annotations

from pathlib import Path

import httpx
from distillation.config.models import GenerationConfig, TeacherConfig
from distillation.teacher.base import TeacherResponse
from distillation.teacher.ollama_client import OllamaTeacherClient


def _client(handler: httpx.MockTransport, **gen: object) -> OllamaTeacherClient:
    cfg = TeacherConfig(model="qwen3.5:2b", generation=GenerationConfig(**gen))
    http = httpx.Client(transport=handler, base_url="http://test")
    return OllamaTeacherClient(cfg, client=http)


def test_think_false_sent_top_level_and_thinking_captured() -> None:
    captured: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {"content": "4", "thinking": "2 plus 2..."},
                "prompt_eval_count": 5,
                "eval_count": 1,
            },
        )

    client = _client(httpx.MockTransport(handle), think=False)
    resp = client.generate("What is 2+2?")

    # `think` is a top-level field, not a sampling option.
    assert captured["think"] is False
    assert "think" not in captured["options"]  # type: ignore[operator]
    assert isinstance(resp, TeacherResponse)
    assert resp.text == "4"
    assert resp.thinking == "2 plus 2..."


def test_think_omitted_when_not_configured() -> None:
    captured: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "hi"}, "eval_count": 1})

    client = _client(httpx.MockTransport(handle))  # think defaults to None
    client.generate("hello")
    assert "think" not in captured  # None -> field omitted, model default kept


def test_empty_reasoning_response_recorded_as_failure(tmp_path: Path) -> None:
    from distillation.datasets.generator import DatasetGenerator
    from distillation.teacher.base import TeacherClient
    from distillation.utils.io import read_jsonl, write_jsonl

    class _EmptyTeacher(TeacherClient):
        @property
        def model(self) -> str:
            return "qwen3.5:9b"

        def generate(self, prompt: str, *, system: str | None = None) -> TeacherResponse:
            from distillation.datasets.schema import TokenUsage

            # Simulates a reasoning model that spent its budget thinking.
            return TeacherResponse(
                text="", usage=TokenUsage(completion_tokens=128), latency_s=0.5, thinking="..."
            )

        def health_check(self) -> bool:
            return True

    prompts = tmp_path / "p.jsonl"
    out = tmp_path / "gen.jsonl"
    write_jsonl(prompts, [{"id": 1, "prompt": "q"}])

    summary = DatasetGenerator(_EmptyTeacher()).run(prompts, out)

    assert summary.failed == 1
    assert summary.generated == 0
    row = next(iter(read_jsonl(out)))
    assert row["error"] is not None
    assert "reasoning model" in row["error"]
    assert row["thinking"] == "..."
