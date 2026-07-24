"""Training-sample format adapters (Phase 2).

A :class:`GenerationRecord` (raw teacher output) is converted into a
training-ready sample by a :class:`FormatAdapter`. Three formats are supported,
each behind the same interface so the builder and the CLI never branch on the
concrete format (Open/Closed principle):

* ``instruction`` — Alpaca-style ``{instruction, input, output}``.
* ``chat``        — role-tagged ``{messages: [...]}`` (mlx-lm / HF chat format).
* ``completion``  — plain ``{prompt, completion}`` pairs.

Adding a new format later means adding one adapter + one registry entry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from distillation.config.models import DatasetFormat
from distillation.datasets.schema import GenerationRecord


# ---------------------------------------------------------------------------
# Typed output records — one per format. Validating on the way out guarantees a
# well-formed training file even if an adapter is later modified.
# ---------------------------------------------------------------------------
class InstructionSample(BaseModel):
    """Alpaca-style supervised sample."""

    instruction: str
    input: str = ""
    output: str


class ChatMessage(BaseModel):
    """A single role-tagged chat turn."""

    role: str
    content: str


class ChatSample(BaseModel):
    """Chat-format sample consumed by mlx-lm / HF chat templates."""

    messages: list[ChatMessage]


class CompletionSample(BaseModel):
    """Prompt/completion pair for causal-LM completion training."""

    prompt: str
    completion: str


# ---------------------------------------------------------------------------
# Adapter interface + implementations.
# ---------------------------------------------------------------------------
class FormatAdapter(ABC):
    """Convert a teacher :class:`GenerationRecord` into a training sample.

    Args:
        system_prompt: Optional default system prompt applied when a record does
            not carry its own. Formats that have no notion of a system message
            (instruction/completion) ignore it.
    """

    def __init__(self, *, system_prompt: str | None = None) -> None:
        self._system_prompt = system_prompt

    @property
    @abstractmethod
    def format(self) -> DatasetFormat:
        """The format this adapter produces."""

    @abstractmethod
    def convert(self, record: GenerationRecord) -> dict[str, Any]:
        """Return a JSON-serializable training sample for ``record``."""

    def _effective_system(self, record: GenerationRecord) -> str | None:
        """Per-record system prompt wins over the adapter default."""
        return record.system or self._system_prompt


class InstructionAdapter(FormatAdapter):
    """Produce Alpaca-style ``{instruction, input, output}`` samples."""

    @property
    def format(self) -> DatasetFormat:
        """The instruction format."""
        return DatasetFormat.INSTRUCTION

    def convert(self, record: GenerationRecord) -> dict[str, Any]:
        """Map the record to an ``{instruction, input, output}`` sample."""
        return InstructionSample(
            instruction=record.prompt,
            input="",
            output=record.response,
        ).model_dump()


class ChatAdapter(FormatAdapter):
    """Produce role-tagged ``{messages: [...]}`` samples."""

    @property
    def format(self) -> DatasetFormat:
        """The chat format."""
        return DatasetFormat.CHAT

    def convert(self, record: GenerationRecord) -> dict[str, Any]:
        """Map the record to a ``{messages: [...]}`` sample."""
        messages: list[ChatMessage] = []
        system = self._effective_system(record)
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=record.prompt))
        messages.append(ChatMessage(role="assistant", content=record.response))
        return ChatSample(messages=messages).model_dump()


class CompletionAdapter(FormatAdapter):
    """Produce ``{prompt, completion}`` pairs."""

    @property
    def format(self) -> DatasetFormat:
        """The completion format."""
        return DatasetFormat.COMPLETION

    def convert(self, record: GenerationRecord) -> dict[str, Any]:
        """Map the record to a ``{prompt, completion}`` sample."""
        return CompletionSample(
            prompt=record.prompt,
            completion=record.response,
        ).model_dump()


# ---------------------------------------------------------------------------
# Registry / factory.
# ---------------------------------------------------------------------------
_ADAPTERS: dict[DatasetFormat, type[FormatAdapter]] = {
    DatasetFormat.INSTRUCTION: InstructionAdapter,
    DatasetFormat.CHAT: ChatAdapter,
    DatasetFormat.COMPLETION: CompletionAdapter,
}


def get_adapter(fmt: DatasetFormat, *, system_prompt: str | None = None) -> FormatAdapter:
    """Instantiate the adapter for a given format.

    Args:
        fmt: Desired output format.
        system_prompt: Default system prompt for formats that support one.

    Returns:
        A ready-to-use :class:`FormatAdapter`.

    Raises:
        KeyError: If no adapter is registered for ``fmt`` (should be unreachable
            given the enum, but guards future additions).
    """
    try:
        adapter_cls = _ADAPTERS[fmt]
    except KeyError as exc:  # pragma: no cover - defensive.
        raise KeyError(f"No format adapter registered for {fmt!r}.") from exc
    return adapter_cls(system_prompt=system_prompt)
