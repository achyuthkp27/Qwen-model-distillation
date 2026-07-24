"""Student-model backend abstraction (Phase 3).

The student can be trained/served by either **MLX** (Apple-Silicon-native,
default) or **HuggingFace PEFT** (PyTorch/MPS). Both implement
:class:`StudentBackend`, so the trainer, evaluator, and inference engine depend
only on this interface (Dependency Inversion) and never import a framework.

Gradient accumulation is orchestrated by the *trainer*, not the backend: the
trainer calls :meth:`accumulate_gradients` for each micro-batch, then
:meth:`apply_gradients` once per optimizer step. This keeps the accumulation
policy in one place and makes the loop trivially testable with a fake backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from distillation.config.models import GenerationConfig, LoRAConfig, TrainingConfig
from distillation.losses.base import DistillationStrategy


@dataclass(slots=True)
class EncodedSample:
    """A tokenized training example.

    Attributes:
        input_ids: Token ids for the full sequence (prompt + completion).
        labels: Next-token targets aligned to ``input_ids``.
        loss_mask: 1 where the token is supervised (completion), 0 elsewhere
            (prompt/padding), so the prompt does not contribute to the loss.
    """

    input_ids: list[int]
    labels: list[int]
    loss_mask: list[int]

    def __len__(self) -> int:
        return len(self.input_ids)


@dataclass(slots=True)
class StepMetrics:
    """Metrics from a single micro-batch forward/backward pass.

    Attributes:
        loss: Total scalar loss for the micro-batch.
        components: Named sub-losses (e.g. ``{"ce": ...}``) for logging.
        num_tokens: Supervised tokens in the micro-batch (for throughput).
    """

    loss: float
    components: dict[str, float] = field(default_factory=dict)
    num_tokens: int = 0


@dataclass(slots=True)
class GenerationResult:
    """Result of a student generation (mirrors :class:`TeacherResponse`)."""

    text: str
    num_tokens: int
    latency_s: float

    @property
    def tokens_per_sec(self) -> float:
        """Generated tokens per second, or 0 when unmeasurable."""
        if self.latency_s <= 0 or self.num_tokens <= 0:
            return 0.0
        return self.num_tokens / self.latency_s


class StudentBackend(ABC):
    """Interface implemented by every student backend (MLX, PEFT)."""

    #: Reasoning control for training tokenization (reasoning models like Qwen3).
    #: None = model default; False = train in think-free format (matches think-off
    #: generation). Set via :meth:`set_thinking`.
    _encode_thinking: bool | None = None

    def set_thinking(self, enable_thinking: bool | None) -> None:
        """Set the reasoning mode used when tokenizing training examples."""
        self._encode_thinking = enable_thinking

    # -- lifecycle ----------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load the base model + tokenizer into memory."""

    @abstractmethod
    def apply_lora(self, lora: LoRAConfig) -> None:
        """Attach LoRA/QLoRA adapters, freezing the base weights."""

    # -- data ---------------------------------------------------------------
    @abstractmethod
    def encode(self, sample: dict[str, Any], max_seq_length: int) -> EncodedSample:
        """Tokenize one formatted training sample into ids/labels/mask.

        The backend applies its own chat template and masks prompt tokens so only
        the teacher completion is supervised.
        """

    @abstractmethod
    def collate(self, samples: list[EncodedSample]) -> Any:
        """Pack encoded samples into a backend-native, padded batch."""

    # -- optimization -------------------------------------------------------
    @abstractmethod
    def create_optimizer(self, training: TrainingConfig) -> None:
        """Initialize the optimizer over the trainable (LoRA) parameters."""

    @abstractmethod
    def accumulate_gradients(self, batch: Any, strategy: DistillationStrategy[Any]) -> StepMetrics:
        """Forward + backward on one micro-batch, accumulating gradients.

        The loss is computed by ``strategy`` using this backend's array ops, so
        the same strategy code differentiates under any backend.
        """

    @abstractmethod
    def apply_gradients(self, learning_rate: float, max_grad_norm: float | None) -> float:
        """Apply accumulated gradients with one optimizer step.

        Returns:
            The global gradient norm (pre-clipping) for logging.
        """

    @abstractmethod
    def zero_gradients(self) -> None:
        """Reset the gradient accumulator after an optimizer step."""

    @abstractmethod
    def eval_loss(self, batch: Any, strategy: DistillationStrategy[Any]) -> StepMetrics:
        """Compute the loss on a batch *without* gradients (validation)."""

    # -- inference ----------------------------------------------------------
    @abstractmethod
    def generate(self, prompt: str, generation: GenerationConfig) -> GenerationResult:
        """Generate a completion from the (possibly adapted) student."""

    # -- persistence --------------------------------------------------------
    @abstractmethod
    def save_adapter(self, path: Path) -> None:
        """Persist the LoRA adapter weights to ``path``."""

    @abstractmethod
    def load_adapter(self, path: Path) -> None:
        """Load previously saved LoRA adapter weights from ``path``."""

    # -- introspection ------------------------------------------------------
    @abstractmethod
    def num_trainable_params(self) -> int:
        """Number of trainable (LoRA) parameters."""

    @abstractmethod
    def memory_mb(self) -> float:
        """Approximate model memory footprint in MiB."""

    def close(self) -> None:  # noqa: B027 - optional hook, intentionally not abstract.
        """Release model resources. Optional."""
