"""MLX student backend (Phase 3) — Apple-Silicon-native LoRA/QLoRA.

Implements :class:`StudentBackend` using ``mlx`` + ``mlx-lm``. All heavy imports
are performed lazily inside :meth:`load` so this module (and the rest of the
framework) imports fine on machines without MLX installed — the dependency is
only required when you actually train/serve with the MLX backend.

Notes on the MLX training model:
* Only LoRA parameters are trainable after :meth:`apply_lora` freezes the base.
* Gradients are functional pytrees; accumulation sums them across micro-batches
  via ``tree_map`` and the optimizer step is applied once per accumulation.
* The distillation loss is computed by the strategy against :class:`MLXOps`, so
  the identical strategy code differentiates under MLX autograd.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from distillation.config.models import GenerationConfig, LoRAConfig, TrainingConfig
from distillation.losses.base import DistillationStrategy, LossInputs
from distillation.student.base import (
    EncodedSample,
    GenerationResult,
    StepMetrics,
    StudentBackend,
)
from distillation.student.tokenization import encode_sample
from distillation.utils.logging import get_logger
from distillation.utils.timing import Stopwatch

_logger = get_logger(__name__)


class MLXOps:
    """``ArrayOps`` implementation over ``mlx.core`` (with autograd)."""

    def __init__(self, mx_module: Any) -> None:
        self._mx = mx_module

    def log_softmax(self, x: Any) -> Any:
        """Stable log-softmax over the last axis."""
        mx = self._mx
        return x - mx.logsumexp(x, axis=-1, keepdims=True)

    def softmax(self, x: Any) -> Any:
        """Softmax over the last axis."""
        return self._mx.softmax(x, axis=-1)

    def gather_last(self, x: Any, idx: Any) -> Any:
        """Gather along the last axis at integer indices."""
        mx = self._mx
        return mx.take_along_axis(x, idx[..., None], axis=-1)[..., 0]

    def masked_mean(self, x: Any, mask: Any) -> Any:
        """Mean over masked (non-zero) positions, safe against empty masks."""
        mx = self._mx
        total = mx.sum(x * mask)
        count = mx.maximum(mx.sum(mask), 1.0)
        return total / count

    def sum_last(self, x: Any) -> Any:
        """Sum over the last axis."""
        return self._mx.sum(x, axis=-1)

    def scalar_mul(self, x: Any, s: float) -> Any:
        """Multiply by a scalar."""
        return x * s

    def add(self, a: Any, b: Any) -> Any:
        """Element-wise addition."""
        return a + b

    def sub(self, a: Any, b: Any) -> Any:
        """Element-wise subtraction."""
        return a - b

    def mul(self, a: Any, b: Any) -> Any:
        """Element-wise multiplication."""
        return a * b

    def neg(self, x: Any) -> Any:
        """Element-wise negation."""
        return -x

    def to_float(self, x: Any) -> float:
        """Materialize a scalar array (evaluates the lazy graph)."""
        return float(x.item())


class MLXStudentBackend(StudentBackend):
    """LoRA fine-tuning + generation for a student model via MLX.

    Args:
        model_id: MLX hub id or local path of the base model.
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model: Any = None
        self._tokenizer: Any = None
        self._optimizer: Any = None
        self._grad_accum: Any = None
        self._mx: Any = None
        self._nn: Any = None
        self._ops: MLXOps | None = None

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        """Load base model + tokenizer and initialize MLX handles."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        self._mx = mx
        self._nn = nn
        self._ops = MLXOps(mx)
        _logger.info("Loading MLX student '%s'...", self._model_id)
        self._model, self._tokenizer = load(self._model_id)

    def apply_lora(self, lora: LoRAConfig) -> None:
        """Freeze the base model and attach LoRA layers."""
        from mlx_lm.tuner.utils import linear_to_lora_layers

        if self._model is None:
            raise RuntimeError("Call load() before apply_lora().")
        self._model.freeze()
        num_layers = lora.num_layers if lora.num_layers >= 0 else len(self._model.layers)
        lora_params = {
            "rank": lora.rank,
            "scale": lora.alpha / lora.rank,
            "dropout": lora.dropout,
        }
        linear_to_lora_layers(self._model, num_layers, lora_params)
        _logger.info(
            "Applied LoRA (rank=%d, alpha=%.1f) to %d layers; trainable params=%d",
            lora.rank,
            lora.alpha,
            num_layers,
            self.num_trainable_params(),
        )

    # -- data ---------------------------------------------------------------
    def encode(self, sample: dict[str, Any], max_seq_length: int) -> EncodedSample:
        """Tokenize a sample, masking prompt tokens out of the loss."""
        return encode_sample(self._tokenizer, sample, max_seq_length, self._encode_thinking)

    def collate(self, samples: list[EncodedSample]) -> dict[str, Any]:
        """Right-pad a list of samples into batched MLX arrays."""
        mx = self._mx
        max_len = max(len(s) for s in samples)

        def pad(seq: list[int]) -> list[int]:
            return seq + [0] * (max_len - len(seq))

        return {
            "input_ids": mx.array([pad(s.input_ids) for s in samples]),
            "labels": mx.array([pad(s.labels) for s in samples]),
            "loss_mask": mx.array([pad(s.loss_mask) for s in samples], dtype=mx.float32),
        }

    # -- optimization -------------------------------------------------------
    def create_optimizer(self, training: TrainingConfig) -> None:
        """Create an AdamW optimizer over the trainable LoRA parameters."""
        import mlx.optimizers as optim

        self._optimizer = optim.AdamW(
            learning_rate=training.learning_rate, weight_decay=training.weight_decay
        )

    def _loss_value(
        self, model: Any, batch: dict[str, Any], strategy: DistillationStrategy[Any]
    ) -> Any:
        """Compute the scalar distillation loss for autograd."""
        assert self._ops is not None, "load() must be called before training."
        logits = model(batch["input_ids"])
        inputs = LossInputs(
            student_logits=logits,
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
        )
        return strategy.compute_loss(inputs, self._ops).total

    def accumulate_gradients(
        self, batch: dict[str, Any], strategy: DistillationStrategy[Any]
    ) -> StepMetrics:
        """Run forward+backward on one micro-batch, summing gradients."""
        from mlx.utils import tree_map

        loss_and_grad = self._nn.value_and_grad(
            self._model, lambda m: self._loss_value(m, batch, strategy)
        )
        loss, grads = loss_and_grad(self._model)
        self._mx.eval(loss, grads)  # force computation

        if self._grad_accum is None:
            self._grad_accum = grads
        else:
            self._grad_accum = tree_map(lambda a, b: a + b, self._grad_accum, grads)
        num_tokens = int(self._mx.sum(batch["loss_mask"]).item())
        return StepMetrics(loss=float(loss.item()), components={}, num_tokens=num_tokens)

    def apply_gradients(self, learning_rate: float, max_grad_norm: float | None) -> float:
        """Clip and apply accumulated gradients; return the pre-clip grad norm."""
        from mlx.utils import tree_map

        mx = self._mx
        self._optimizer.learning_rate = learning_rate
        grads = self._grad_accum

        # Global L2 norm across the gradient pytree.
        leaves = [g for _, g in _tree_leaves(grads)]
        sq = sum(mx.sum(g * g) for g in leaves) if leaves else mx.array(0.0)
        grad_norm = float(mx.sqrt(sq).item())

        if max_grad_norm is not None and grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-6)
            grads = tree_map(lambda g: g * scale, grads)

        self._optimizer.update(self._model, grads)
        mx.eval(self._model.parameters(), self._optimizer.state)
        return grad_norm

    def zero_gradients(self) -> None:
        """Discard the accumulated gradient pytree."""
        self._grad_accum = None

    def eval_loss(self, batch: dict[str, Any], strategy: DistillationStrategy[Any]) -> StepMetrics:
        """Compute validation loss without gradients."""
        loss = self._loss_value(self._model, batch, strategy)
        self._mx.eval(loss)
        num_tokens = int(self._mx.sum(batch["loss_mask"]).item())
        return StepMetrics(loss=float(loss.item()), components={}, num_tokens=num_tokens)

    # -- inference ----------------------------------------------------------
    def generate(self, prompt: str, generation: GenerationConfig) -> GenerationResult:
        """Generate a completion with the current student weights."""
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        messages = []
        if generation.system_prompt:
            messages.append({"role": "system", "content": generation.system_prompt})
        messages.append({"role": "user", "content": prompt})
        # Reasoning models (Qwen3) emit a <think> stream that can eat the whole
        # token budget. Reuse the same `think` config as the teacher: when set,
        # pass enable_thinking to the chat template (ignored by non-reasoning models).
        template_kwargs: dict[str, Any] = {}
        if generation.think is not None:
            template_kwargs["enable_thinking"] = generation.think
        text_prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **template_kwargs
        )
        sampler = make_sampler(temp=generation.temperature, top_p=generation.top_p)
        with Stopwatch() as sw:
            text = mlx_generate(
                self._model,
                self._tokenizer,
                prompt=text_prompt,
                max_tokens=generation.max_tokens,
                sampler=sampler,
                verbose=False,
            )
        num_tokens = len(self._tokenizer.encode(text))
        return GenerationResult(text=text, num_tokens=num_tokens, latency_s=sw.elapsed_s)

    # -- persistence --------------------------------------------------------
    @staticmethod
    def _resolve(path: Path) -> Path:
        """MLX convention: a single ``.safetensors`` file for the adapter."""
        return path if path.suffix == ".safetensors" else path.with_suffix(".safetensors")

    def save_adapter(self, path: Path) -> None:
        """Save trainable LoRA weights as safetensors."""
        from mlx.utils import tree_flatten

        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        adapter = dict(tree_flatten(self._model.trainable_parameters()))
        self._mx.save_safetensors(str(target), adapter)
        _logger.info("Saved adapter -> %s", target)

    def load_adapter(self, path: Path) -> None:
        """Load LoRA weights from safetensors into the model."""
        from mlx.utils import tree_unflatten

        target = self._resolve(path)
        weights = list(self._mx.load(str(target)).items())
        self._model.update(tree_unflatten(weights))
        _logger.info("Loaded adapter <- %s", target)

    # -- introspection ------------------------------------------------------
    def num_trainable_params(self) -> int:
        """Count trainable (LoRA) parameters."""
        from mlx.utils import tree_flatten

        if self._model is None:
            return 0
        return sum(v.size for _, v in tree_flatten(self._model.trainable_parameters()))

    def memory_mb(self) -> float:
        """Peak MLX GPU memory in MiB (0 if unavailable)."""
        if self._mx is None:
            return 0.0
        try:
            return float(self._mx.get_peak_memory()) / (1024 * 1024)
        except AttributeError:  # older MLX
            return 0.0


def _tree_leaves(tree: Any) -> list[tuple[str, Any]]:
    """Flatten an MLX gradient pytree to (path, array) pairs."""
    from mlx.utils import tree_flatten

    leaves: list[tuple[str, Any]] = tree_flatten(tree)
    return leaves
