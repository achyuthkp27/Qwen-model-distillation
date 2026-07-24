"""Local LLM distillation framework, optimized for Apple Silicon.

The package is organized by responsibility so that new distillation strategies
(logit KL, hidden-state, attention) can be added without touching the trainer:

- ``config``      : Pydantic-validated configuration models + YAML loader.
- ``utils``       : Cross-cutting helpers (logging, timing, memory, seeding, IO).
- ``teacher``     : Teacher-model clients (currently Ollama).
- ``datasets``    : Dataset generation (Phase 1) and format adapters (Phase 2).
- ``student``     : Student-model backends behind a common interface (MLX / PEFT).
- ``trainers``    : Training loops, gradient accumulation, checkpointing.
- ``losses``      : Distillation strategies (response / logit / hidden / attention).
- ``evaluation``  : Metrics, benchmarking, and HTML reporting.
- ``inference``   : CLI, REST API, batch, and interactive chat.
- ``scripts``     : Thin CLI entrypoints wiring the above together.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
