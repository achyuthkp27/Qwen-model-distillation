"""Pydantic configuration models.

Design notes
------------
* Every model sets ``extra="forbid"`` so typos in YAML fail loudly instead of
  being silently ignored.
* Enums are used for closed choice sets (dataset format, backend, strategy) so
  invalid values are rejected at load time rather than deep inside a trainer.
* Sub-configs are intentionally granular: adding a future strategy (logit KL,
  hidden-state) means adding fields here, not rewriting the trainer.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator


class _StrictModel(BaseModel):
    """Base model that forbids unknown fields to catch config typos early."""

    model_config = {"extra": "forbid", "frozen": False}


# ---------------------------------------------------------------------------
# Enumerations.
# ---------------------------------------------------------------------------
class DatasetFormat(StrEnum):
    """Supported training-sample serialization formats (Phase 2)."""

    INSTRUCTION = "instruction"
    CHAT = "chat"
    COMPLETION = "completion"


class StudentBackendKind(StrEnum):
    """Selectable student training/inference backends (Phase 3)."""

    MLX = "mlx"
    PEFT = "peft"


class DistillationStrategyKind(StrEnum):
    """Distillation strategies (Phase 4).

    Only ``RESPONSE`` is implemented today; the rest are declared now so configs
    and the strategy registry are forward-compatible.
    """

    RESPONSE = "response"
    LOGIT = "logit"
    HIDDEN_STATE = "hidden_state"
    ATTENTION = "attention"


# ---------------------------------------------------------------------------
# Phase 1 — Teacher + generation.
# ---------------------------------------------------------------------------
class OllamaConfig(_StrictModel):
    """Connection settings for a local Ollama server."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama base URL.")
    request_timeout_s: PositiveFloat = Field(default=600.0, description="Per-request timeout.")
    max_retries: PositiveInt = Field(default=3, description="Retry attempts on transient errors.")
    max_concurrency: PositiveInt = Field(
        default=1,
        description="Concurrent generation requests. Keep low on unified memory.",
    )


class GenerationConfig(_StrictModel):
    """Sampling parameters for teacher generation."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: PositiveInt = Field(default=40)
    max_tokens: PositiveInt = Field(default=1024, description="Max tokens to generate.")
    seed: int | None = Field(default=None, description="Per-request sampling seed, if fixed.")
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt prepended to every request."
    )
    think: bool | None = Field(
        default=None,
        description=(
            "Reasoning control for reasoning-capable Ollama models (e.g. qwen3.5). "
            "Set False to disable the <think> stream and return the final answer "
            "directly (recommended for distillation teachers); True to force it; "
            "None to use the model's default. Ignored by non-Ollama backends."
        ),
    )


class TeacherConfig(_StrictModel):
    """Teacher model definition (Phase 1)."""

    model: str = Field(default="qwen3:8b", description="Ollama model tag for the teacher.")
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


# ---------------------------------------------------------------------------
# Phase 1/2 — Dataset.
# ---------------------------------------------------------------------------
class DatasetConfig(_StrictModel):
    """Dataset paths and formatting (Phases 1 & 2)."""

    prompts_path: Path = Field(description="Input JSONL of prompts ({id, prompt}).")
    generations_path: Path = Field(
        default=Path("outputs/teacher_generations.jsonl"),
        description="Where raw teacher generations are written.",
    )
    train_path: Path = Field(
        default=Path("outputs/train.jsonl"),
        description="Formatted training set output.",
    )
    val_path: Path | None = Field(
        default=Path("outputs/val.jsonl"),
        description="Optional formatted validation set output.",
    )
    format: DatasetFormat = Field(default=DatasetFormat.CHAT)
    val_fraction: float = Field(default=0.1, ge=0.0, lt=1.0, description="Held-out val fraction.")
    shuffle: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Phase 3 — Student + training.
# ---------------------------------------------------------------------------
class LoRAConfig(_StrictModel):
    """LoRA / QLoRA adapter hyperparameters (Phase 3)."""

    rank: PositiveInt = Field(default=16)
    alpha: PositiveFloat = Field(default=32.0)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    # Which projection modules to adapt; backend maps these to concrete layers.
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    num_layers: int = Field(
        default=-1,
        description="Number of transformer layers to apply LoRA to (-1 = all).",
    )
    quantize: bool = Field(default=False, description="Enable QLoRA (quantized base weights).")


class StudentConfig(_StrictModel):
    """Student model definition (Phase 3)."""

    model: str = Field(
        default="mlx-community/Qwen3-4B-4bit",
        description="Student base model (MLX hub id or HF repo depending on backend).",
    )
    backend: StudentBackendKind = Field(default=StudentBackendKind.MLX)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)


class TrainingConfig(_StrictModel):
    """Optimization / loop settings (Phase 3)."""

    learning_rate: PositiveFloat = Field(default=1e-4)
    epochs: PositiveInt = Field(default=3)
    batch_size: PositiveInt = Field(default=1, description="Micro-batch size.")
    gradient_accumulation_steps: PositiveInt = Field(default=8)
    max_seq_length: PositiveInt = Field(default=2048)
    warmup_steps: int = Field(default=0, ge=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: PositiveFloat | None = Field(default=1.0)
    checkpoint_interval: PositiveInt = Field(
        default=200, description="Save an adapter checkpoint every N optimizer steps."
    )
    eval_interval: PositiveInt = Field(default=200)
    log_interval: PositiveInt = Field(default=10)

    @property
    def effective_batch_size(self) -> int:
        """Global batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


# ---------------------------------------------------------------------------
# Phase 4 — Distillation strategy.
# ---------------------------------------------------------------------------
class DistillationConfig(_StrictModel):
    """Which distillation strategy to run, plus its hyperparameters (Phase 4).

    Fields for not-yet-implemented strategies are present so the architecture is
    stable: enabling logit distillation later is a config change, not a rewrite.
    """

    strategy: DistillationStrategyKind = Field(default=DistillationStrategyKind.RESPONSE)
    # Logit/KL distillation knobs (reserved for future use).
    temperature: PositiveFloat = Field(
        default=2.0, description="Softmax temperature for logit distillation."
    )
    alpha_ce: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight of hard-label cross-entropy."
    )
    alpha_kd: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Weight of the distillation term."
    )

    @model_validator(mode="after")
    def _check_weights(self) -> DistillationConfig:
        if self.strategy is not DistillationStrategyKind.RESPONSE:
            total = self.alpha_ce + self.alpha_kd
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"alpha_ce + alpha_kd must sum to 1.0, got {total}.")
        return self


# ---------------------------------------------------------------------------
# Phase 5 — Evaluation.
# ---------------------------------------------------------------------------
class EvaluationConfig(_StrictModel):
    """Evaluation and reporting settings (Phase 5)."""

    metrics: list[str] = Field(
        default_factory=lambda: ["bleu", "rouge", "bertscore", "latency", "tokens_per_sec"]
    )
    num_samples: PositiveInt = Field(default=100, description="Eval prompts to sample.")
    eval_prompts_path: Path | None = Field(
        default=None,
        description=(
            "Optional held-out prompt JSONL used for evaluation. If set, avoids "
            "measuring on training prompts (leakage). Falls back to "
            "dataset.prompts_path when unset."
        ),
    )
    bertscore_model: str = Field(default="microsoft/deberta-xlarge-mnli")
    report_path: Path = Field(default=Path("outputs/report.html"))


# ---------------------------------------------------------------------------
# Top-level experiment.
# ---------------------------------------------------------------------------
class ExperimentConfig(_StrictModel):
    """Root configuration object for a single experiment.

    One YAML file == one :class:`ExperimentConfig`. The ``name`` determines the
    experiment directory under ``experiments/`` where configs, logs, metrics,
    plots, and adapters are stored (Phase 7).
    """

    name: str = Field(description="Unique experiment name, e.g. 'experiment_001'.")
    seed: int = Field(default=42)
    output_root: Path = Field(
        default=Path("experiments"),
        description="Root directory under which per-experiment artifacts are stored.",
    )

    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    dataset: DatasetConfig
    student: StudentConfig = Field(default_factory=StudentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def experiment_dir(self) -> Path:
        """Directory holding all artifacts for this experiment (Phase 7)."""
        return self.output_root / self.name
