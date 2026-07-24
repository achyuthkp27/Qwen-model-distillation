"""REST inference API (Phase 6).

A minimal FastAPI application exposing the student for HTTP generation. FastAPI /
Pydantic-v2 request-response models keep it typed. Imported lazily via
:func:`create_app` so the framework does not require the ``serve`` extra unless
you actually run the server.
"""
# FastAPI's route decorators are untyped (optional dep resolved to Any); this
# module is thin glue, so we silence the resulting decorator noise here.
# mypy: disable-error-code="untyped-decorator"

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from distillation.inference.engine import InferenceEngine

if TYPE_CHECKING:
    from fastapi import FastAPI


class GenerateRequest(BaseModel):
    """A single generation request."""

    prompt: str = Field(description="The user prompt.")


class BatchRequest(BaseModel):
    """A batch generation request."""

    prompts: list[str] = Field(description="Prompts to generate for.")


class GenerateResponse(BaseModel):
    """A single generation response with performance fields."""

    text: str
    num_tokens: int
    latency_s: float
    tokens_per_sec: float


def create_app(engine: InferenceEngine) -> FastAPI:
    """Build a FastAPI app serving the given inference engine.

    Args:
        engine: A configured (not necessarily loaded) inference engine.

    Returns:
        A FastAPI application with ``/health``, ``/generate``, ``/generate/batch``.

    Raises:
        ImportError: If FastAPI is not installed (``pip install -e '.[serve]'``).
    """
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - env-dependent.
        raise ImportError("REST API requires: pip install -e '.[serve]'") from exc

    app = FastAPI(title="Distillation student API", version="0.1.0")

    @app.on_event("startup")
    def _startup() -> None:
        engine.load()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        result = engine.generate(req.prompt)
        return GenerateResponse(
            text=result.text,
            num_tokens=result.num_tokens,
            latency_s=result.latency_s,
            tokens_per_sec=result.tokens_per_sec,
        )

    @app.post("/generate/batch", response_model=list[GenerateResponse])
    def generate_batch(req: BatchRequest) -> list[GenerateResponse]:
        return [
            GenerateResponse(
                text=r.text,
                num_tokens=r.num_tokens,
                latency_s=r.latency_s,
                tokens_per_sec=r.tokens_per_sec,
            )
            for r in engine.batch(req.prompts)
        ]

    return app
