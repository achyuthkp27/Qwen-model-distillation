"""Dataset generation (Phase 1) and format adapters (Phase 2)."""

from __future__ import annotations

from distillation.datasets.builder import BuildSummary, DatasetBuilder
from distillation.datasets.formats import (
    ChatAdapter,
    CompletionAdapter,
    FormatAdapter,
    InstructionAdapter,
    get_adapter,
)
from distillation.datasets.generator import DatasetGenerator, GenerationSummary
from distillation.datasets.schema import (
    GenerationRecord,
    PromptRecord,
    TokenUsage,
)

__all__ = [
    "BuildSummary",
    "ChatAdapter",
    "CompletionAdapter",
    "DatasetBuilder",
    "DatasetGenerator",
    "FormatAdapter",
    "GenerationRecord",
    "GenerationSummary",
    "InstructionAdapter",
    "PromptRecord",
    "TokenUsage",
    "get_adapter",
]
