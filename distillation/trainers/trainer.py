"""Distillation training loop (Phase 3).

Backend-agnostic orchestration: epoch iteration, micro-batching, gradient
accumulation, LR scheduling, gradient clipping, periodic checkpointing, periodic
validation, and metrics recording. It drives any :class:`StudentBackend` and any
:class:`DistillationStrategy`, so switching MLX<->PEFT or response<->logit
distillation never touches this file.

Every optimizer step corresponds to ``gradient_accumulation_steps`` micro-batches
of ``batch_size`` samples each (effective batch = their product).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillation.config.models import TrainingConfig
from distillation.losses.base import DistillationStrategy
from distillation.student.base import EncodedSample, StudentBackend
from distillation.trainers.metrics import MetricRecord, MetricsRecorder
from distillation.trainers.schedule import warmup_constant
from distillation.utils.logging import get_logger
from distillation.utils.timing import Stopwatch

_logger = get_logger(__name__)


@dataclass(slots=True)
class TrainResult:
    """Summary of a completed training run.

    Attributes:
        total_steps: Optimizer steps performed.
        final_train_loss: Loss of the last logged train step.
        best_val_loss: Best validation loss (``None`` if no validation set).
        adapter_path: Path of the final saved adapter.
        metrics_path: Path of the metrics JSONL.
    """

    total_steps: int
    final_train_loss: float
    best_val_loss: float | None
    adapter_path: Path
    metrics_path: Path


class DistillationTrainer:
    """Runs LoRA distillation training over a formatted dataset.

    Args:
        backend: A loaded backend with LoRA applied and optimizer created.
        strategy: The distillation objective.
        training: Optimization hyperparameters.
        recorder: Metrics sink.
        adapter_dir: Directory where adapter checkpoints are written.
    """

    def __init__(
        self,
        backend: StudentBackend,
        strategy: DistillationStrategy[Any],
        training: TrainingConfig,
        *,
        recorder: MetricsRecorder,
        adapter_dir: Path,
    ) -> None:
        self._backend = backend
        self._strategy = strategy
        self._cfg = training
        self._recorder = recorder
        self._adapter_dir = adapter_dir
        self._schedule = warmup_constant(training.learning_rate, training.warmup_steps)

    @staticmethod
    def _batches(samples: list[EncodedSample], batch_size: int) -> list[list[EncodedSample]]:
        """Split samples into contiguous micro-batches of ``batch_size``."""
        return [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]

    def _encode_all(self, raw_samples: list[dict[str, Any]]) -> list[EncodedSample]:
        """Tokenize every raw sample once up front."""
        return [self._backend.encode(s, self._cfg.max_seq_length) for s in raw_samples]

    def _adapter_path(self, step: int | str) -> Path:
        """Base (suffix-less) path for an adapter checkpoint at a given step.

        Each backend appends its own convention (MLX: ``.safetensors`` file;
        PEFT: a directory), so the trainer stays backend-agnostic.
        """
        return self._adapter_dir / f"adapter_{step}"

    def _validate(self, val_batches: list[list[EncodedSample]], step: int, epoch: int) -> None:
        """Run validation and log the mean loss."""
        if not val_batches:
            return
        losses = []
        for micro in val_batches:
            batch = self._backend.collate(micro)
            losses.append(self._backend.eval_loss(batch, self._strategy).loss)
        mean_loss = sum(losses) / len(losses)
        self._recorder.log(
            MetricRecord(
                step=step,
                epoch=epoch,
                split="val",
                loss=mean_loss,
                memory_mb=self._backend.memory_mb(),
            )
        )
        _logger.info("[step %d] val_loss=%.4f", step, mean_loss)

    def train(
        self,
        train_samples: list[dict[str, Any]],
        val_samples: list[dict[str, Any]] | None = None,
    ) -> TrainResult:
        """Execute the full training loop.

        Args:
            train_samples: Formatted training samples (Phase 2 output rows).
            val_samples: Optional formatted validation samples.

        Returns:
            A :class:`TrainResult`.
        """
        train_encoded = self._encode_all(train_samples)
        val_batches = (
            self._batches(self._encode_all(val_samples), self._cfg.batch_size)
            if val_samples
            else []
        )

        accum = self._cfg.gradient_accumulation_steps
        opt_step = 0
        micro_since_step = 0
        final_train_loss = 0.0
        running_loss = 0.0
        running_tokens = 0
        sw = Stopwatch().start()

        _logger.info(
            "Training: %d samples, %d epochs, effective batch %d (%d micro x %d accum)",
            len(train_encoded),
            self._cfg.epochs,
            self._cfg.effective_batch_size,
            self._cfg.batch_size,
            accum,
        )

        for epoch in range(self._cfg.epochs):
            for micro in self._batches(train_encoded, self._cfg.batch_size):
                batch = self._backend.collate(micro)
                metrics = self._backend.accumulate_gradients(batch, self._strategy)
                running_loss += metrics.loss
                running_tokens += metrics.num_tokens
                micro_since_step += 1

                if micro_since_step < accum:
                    continue

                # --- one optimizer step ---
                opt_step += 1
                lr = self._schedule(opt_step)
                grad_norm = self._backend.apply_gradients(lr, self._cfg.max_grad_norm)
                self._backend.zero_gradients()

                avg_loss = running_loss / micro_since_step
                final_train_loss = avg_loss
                elapsed = sw.stop()
                tps = running_tokens / elapsed if elapsed > 0 else 0.0
                sw.start()

                if opt_step % self._cfg.log_interval == 0:
                    self._recorder.log(
                        MetricRecord(
                            step=opt_step,
                            epoch=epoch,
                            split="train",
                            loss=avg_loss,
                            learning_rate=lr,
                            grad_norm=grad_norm,
                            tokens_per_sec=tps,
                            memory_mb=self._backend.memory_mb(),
                        )
                    )
                    _logger.info(
                        "[step %d] loss=%.4f lr=%.2e grad_norm=%.2f %.0f tok/s",
                        opt_step,
                        avg_loss,
                        lr,
                        grad_norm,
                        tps,
                    )

                if opt_step % self._cfg.eval_interval == 0:
                    self._validate(val_batches, opt_step, epoch)

                if opt_step % self._cfg.checkpoint_interval == 0:
                    self._backend.save_adapter(self._adapter_path(opt_step))

                running_loss = 0.0
                running_tokens = 0
                micro_since_step = 0

        # Final validation + checkpoint.
        self._validate(val_batches, opt_step, self._cfg.epochs - 1)
        final_path = self._adapter_path("final")
        self._backend.save_adapter(final_path)
        _logger.info("Training complete: %d steps, final loss %.4f", opt_step, final_train_loss)

        return TrainResult(
            total_steps=opt_step,
            final_train_loss=final_train_loss,
            best_val_loss=self._recorder.best_val_loss(),
            adapter_path=final_path,
            metrics_path=self._recorder.path,
        )
