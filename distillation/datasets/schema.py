"""Data record schemas shared across generation, training, and evaluation.

These Pydantic models define the *on-disk contract* for JSONL artifacts. Using
typed records (rather than bare dicts) means a malformed generation file fails
fast at load time and every downstream phase reads the same well-known fields.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class PromptRecord(BaseModel):
    """A single input prompt (Phase 1 input format).

    Matches the spec example ``{"id": 1, "prompt": "Explain Redis."}`` while
    allowing an optional per-prompt system override and freeform metadata.
    """

    model_config = {"extra": "allow"}  # Tolerate extra user-provided columns.

    id: int | str = Field(description="Stable unique identifier for the prompt.")
    prompt: str = Field(description="The user prompt sent to the teacher.")
    system: str | None = Field(default=None, description="Optional per-prompt system prompt.")


class TokenUsage(BaseModel):
    """Token accounting for a single generation."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class GenerationRecord(BaseModel):
    """A teacher generation plus provenance/metrics (Phase 1 output format).

    This is the canonical unit produced by the generation pipeline and consumed
    by the Phase 2 formatters. It intentionally stores everything needed to
    later reproduce or audit the run.
    """

    id: int | str = Field(description="Prompt id this generation corresponds to.")
    prompt: str
    system: str | None = None
    response: str = Field(description="The teacher's completion text.")

    teacher_model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_s: float = Field(ge=0.0, description="Wall-clock generation latency in seconds.")
    tokens_per_sec: float = Field(default=0.0, ge=0.0)
    thinking: str | None = Field(
        default=None,
        description="Reasoning stream for reasoning models (provenance; not the training target).",
    )

    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="UTC ISO-8601 timestamp of generation.",
    )
    error: str | None = Field(
        default=None, description="Populated if generation failed for this prompt."
    )

    @property
    def ok(self) -> bool:
        """Whether this record represents a successful generation."""
        return self.error is None and bool(self.response)
