"""Inference: engine, REST API, interactive chat, and comparison (Phase 6)."""

from __future__ import annotations

from distillation.inference.chat import run_chat
from distillation.inference.compare import Comparison, compare_once, run_compare_loop
from distillation.inference.engine import InferenceEngine

__all__ = [
    "Comparison",
    "InferenceEngine",
    "compare_once",
    "run_chat",
    "run_compare_loop",
]
