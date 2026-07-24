"""Response (sequence-level) distillation strategy (Phase 4).

The student is trained to reproduce the teacher's *generated text* via standard
masked next-token cross-entropy on the teacher completion. This is the dark-
knowledge-free baseline: no teacher logits are needed, only the teacher's
sampled responses (already produced in Phase 1).

Because the math is written against :class:`ArrayOps`, this identical class runs
under NumPy (tests), MLX, and PyTorch.
"""

from __future__ import annotations

from distillation.config.models import DistillationStrategyKind
from distillation.losses.base import (
    DistillationStrategy,
    LossInputs,
    LossResult,
    register_strategy,
)
from distillation.losses.ops import ArrayOps


@register_strategy(DistillationStrategyKind.RESPONSE)
class ResponseDistillation[A](DistillationStrategy[A]):
    """Masked cross-entropy against teacher-generated tokens.

    Args:
        config: Distillation config (unused fields are ignored; present so every
            strategy shares a uniform constructor signature).
    """

    requires_teacher_logits = False
    requires_hidden_states = False
    requires_attentions = False

    @property
    def kind(self) -> DistillationStrategyKind:
        """The response strategy kind."""
        return DistillationStrategyKind.RESPONSE

    def compute_loss(self, inputs: LossInputs[A], ops: ArrayOps[A]) -> LossResult[A]:
        """Return masked next-token cross-entropy over the completion span.

        ``loss = -mean_{t in completion} log p_student(label_t | x_<t)``
        """
        log_probs = ops.log_softmax(inputs.student_logits)  # [B, T, V]
        target_log_probs = ops.gather_last(log_probs, inputs.labels)  # [B, T]
        # Negative log-likelihood averaged over supervised (masked) positions.
        nll = ops.neg(ops.masked_mean(target_log_probs, inputs.loss_mask))
        # Components stay as arrays (not floats) so this is safe inside autograd.
        return LossResult(total=nll, components={"ce": nll})
