"""Training loop, LR schedules, and metrics recording (Phase 3)."""

from __future__ import annotations

from distillation.trainers.metrics import MetricRecord, MetricsRecorder
from distillation.trainers.schedule import LRSchedule, warmup_constant
from distillation.trainers.trainer import DistillationTrainer, TrainResult

__all__ = [
    "DistillationTrainer",
    "LRSchedule",
    "MetricRecord",
    "MetricsRecorder",
    "TrainResult",
    "warmup_constant",
]
