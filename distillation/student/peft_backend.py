"""HuggingFace PEFT student backend (Phase 3) — PyTorch/MPS LoRA/QLoRA.

The PyTorch counterpart to :mod:`distillation.student.mlx_backend`, behind the
same :class:`StudentBackend` interface. Heavy imports are lazy so the module
loads without torch/transformers/peft installed.

PyTorch accumulates gradients natively (``loss.backward()`` sums into ``.grad``),
so :meth:`accumulate_gradients` does forward+backward, :meth:`apply_gradients`
clips+steps, and :meth:`zero_gradients` calls ``optimizer.zero_grad()``.
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


class TorchOps:
    """``ArrayOps`` implementation over ``torch`` (with autograd)."""

    def __init__(self, torch_module: Any) -> None:
        self._t = torch_module

    def log_softmax(self, x: Any) -> Any:
        """Stable log-softmax over the last axis."""
        return self._t.log_softmax(x, dim=-1)

    def softmax(self, x: Any) -> Any:
        """Softmax over the last axis."""
        return self._t.softmax(x, dim=-1)

    def gather_last(self, x: Any, idx: Any) -> Any:
        """Gather along the last axis at integer indices."""
        return self._t.gather(x, -1, idx.unsqueeze(-1)).squeeze(-1)

    def masked_mean(self, x: Any, mask: Any) -> Any:
        """Mean over masked (non-zero) positions, safe against empty masks."""
        total = (x * mask).sum()
        count = mask.sum().clamp(min=1.0)
        return total / count

    def sum_last(self, x: Any) -> Any:
        """Sum over the last axis."""
        return x.sum(dim=-1)

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
        """Materialize a scalar tensor as a float."""
        return float(x.item())


class PEFTStudentBackend(StudentBackend):
    """LoRA fine-tuning + generation for a student model via HF PEFT.

    Args:
        model_id: HF repo id or local path of the base model.
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model: Any = None
        self._tokenizer: Any = None
        self._optimizer: Any = None
        self._torch: Any = None
        self._ops: TorchOps | None = None
        self._device: str = "cpu"

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        """Load base model + tokenizer onto the best available device."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._ops = TorchOps(torch)
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        _logger.info("Loading PEFT student '%s' on %s...", self._model_id, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # `dtype` replaced the deprecated `torch_dtype` kwarg in Transformers 5.x.
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, dtype=torch.float16).to(
            self._device
        )

    def apply_lora(self, lora: LoRAConfig) -> None:
        """Wrap the base model with a PEFT LoRA adapter."""
        from peft import LoraConfig, get_peft_model

        if self._model is None:
            raise RuntimeError("Call load() before apply_lora().")
        peft_config = LoraConfig(
            r=lora.rank,
            lora_alpha=int(lora.alpha),
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules,
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, peft_config)
        _logger.info("Applied PEFT LoRA; trainable params=%d", self.num_trainable_params())

    # -- data ---------------------------------------------------------------
    def encode(self, sample: dict[str, Any], max_seq_length: int) -> EncodedSample:
        """Tokenize a sample, masking prompt tokens out of the loss."""
        return encode_sample(self._tokenizer, sample, max_seq_length, self._encode_thinking)

    def collate(self, samples: list[EncodedSample]) -> dict[str, Any]:
        """Right-pad samples into batched torch tensors on-device."""
        torch = self._torch
        max_len = max(len(s) for s in samples)

        def pad(seq: list[int]) -> list[int]:
            return seq + [0] * (max_len - len(seq))

        dev = self._device
        # input_ids/labels MUST be int64: embedding lookups and torch.gather
        # require integer indices (MPS rejects float indices outright).
        return {
            "input_ids": torch.tensor(
                [pad(s.input_ids) for s in samples], dtype=torch.long, device=dev
            ),
            "labels": torch.tensor([pad(s.labels) for s in samples], dtype=torch.long, device=dev),
            "loss_mask": torch.tensor(
                [pad(s.loss_mask) for s in samples], dtype=torch.float32, device=dev
            ),
        }

    # -- optimization -------------------------------------------------------
    def create_optimizer(self, training: TrainingConfig) -> None:
        """Create an AdamW optimizer over the trainable LoRA parameters."""
        params = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = self._torch.optim.AdamW(
            params, lr=training.learning_rate, weight_decay=training.weight_decay
        )

    def accumulate_gradients(
        self, batch: dict[str, Any], strategy: DistillationStrategy[Any]
    ) -> StepMetrics:
        """Forward+backward on one micro-batch (grads accumulate in ``.grad``)."""
        assert self._ops is not None, "load() must be called before training."
        self._model.train()
        logits = self._model(input_ids=batch["input_ids"]).logits
        inputs = LossInputs(
            student_logits=logits,
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
        )
        loss = strategy.compute_loss(inputs, self._ops).total
        loss.backward()
        num_tokens = int(batch["loss_mask"].sum().item())
        return StepMetrics(loss=float(loss.item()), components={}, num_tokens=num_tokens)

    def apply_gradients(self, learning_rate: float, max_grad_norm: float | None) -> float:
        """Clip and apply gradients with one optimizer step."""
        for group in self._optimizer.param_groups:
            group["lr"] = learning_rate
        params = [p for p in self._model.parameters() if p.requires_grad]
        if max_grad_norm is not None:
            norm = self._torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            grad_norm = float(norm)
        else:
            grad_norm = 0.0
        self._optimizer.step()
        return grad_norm

    def zero_gradients(self) -> None:
        """Zero the optimizer gradients."""
        self._optimizer.zero_grad()

    def eval_loss(self, batch: dict[str, Any], strategy: DistillationStrategy[Any]) -> StepMetrics:
        """Compute validation loss without gradients."""
        assert self._ops is not None, "load() must be called before training."
        self._model.eval()
        with self._torch.no_grad():
            logits = self._model(input_ids=batch["input_ids"]).logits
            inputs = LossInputs(
                student_logits=logits,
                labels=batch["labels"],
                loss_mask=batch["loss_mask"],
            )
            loss = strategy.compute_loss(inputs, self._ops).total
        num_tokens = int(batch["loss_mask"].sum().item())
        return StepMetrics(loss=float(loss.item()), components={}, num_tokens=num_tokens)

    # -- inference ----------------------------------------------------------
    def generate(self, prompt: str, generation: GenerationConfig) -> GenerationResult:
        """Generate a completion with the current student weights."""
        self._model.eval()
        messages = []
        if generation.system_prompt:
            messages.append({"role": "system", "content": generation.system_prompt})
        messages.append({"role": "user", "content": prompt})
        # Render to a string, then tokenize: apply_chat_template(tokenize=True)
        # returns a BatchEncoding (no .shape) in Transformers 5.x. The string path
        # is stable across versions.
        template_kwargs: dict[str, Any] = {}
        if generation.think is not None:
            template_kwargs["enable_thinking"] = generation.think
        text_prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **template_kwargs
        )
        enc = self._tokenizer(text_prompt, return_tensors="pt").to(self._device)
        prompt_len = enc["input_ids"].shape[1]

        with Stopwatch() as sw, self._torch.no_grad():
            output = self._model.generate(
                **enc,
                max_new_tokens=generation.max_tokens,
                temperature=generation.temperature,
                top_p=generation.top_p,
                do_sample=generation.temperature > 0,
            )
        new_tokens = output[0][prompt_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenerationResult(
            text=text, num_tokens=int(new_tokens.shape[0]), latency_s=sw.elapsed_s
        )

    # -- persistence --------------------------------------------------------
    def save_adapter(self, path: Path) -> None:
        """Save the PEFT adapter directory."""
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))
        _logger.info("Saved adapter -> %s", path)

    def load_adapter(self, path: Path) -> None:
        """Load a PEFT adapter directory into the model."""
        self._model.load_adapter(str(path), adapter_name="default")
        _logger.info("Loaded adapter <- %s", path)

    # -- introspection ------------------------------------------------------
    def num_trainable_params(self) -> int:
        """Count trainable (LoRA) parameters."""
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def memory_mb(self) -> float:
        """Current MPS allocated memory in MiB (0 if unavailable)."""
        if self._torch is None or self._device != "mps":
            return 0.0
        try:
            return float(self._torch.mps.current_allocated_memory()) / (1024 * 1024)
        except AttributeError:
            return 0.0
