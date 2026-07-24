"""Teacher-model clients.

The teacher is abstracted behind :class:`TeacherClient` so that a future
multi-teacher setup, or a non-Ollama backend (vLLM, an HTTP endpoint, or a local
MLX teacher), can be swapped in without changing the generation pipeline.
"""

from __future__ import annotations

from distillation.teacher.base import TeacherClient, TeacherResponse
from distillation.teacher.ollama_client import OllamaTeacherClient

__all__ = ["OllamaTeacherClient", "TeacherClient", "TeacherResponse"]
