"""Distillation-strategy abstraction (Phase 4).

This is the extension point that makes the whole framework future-proof: the
trainer computes gradients over ``strategy.compute_loss(...)`` and never knows
whether it is doing response, logit-KL, hidden-state, or attention distillation.

Adding a new strategy = implement :class:`DistillationStrategy` + register it.
No trainer, backend, or CLI change required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from distillation.config.models import DistillationConfig, DistillationStrategyKind
from distillation.losses.ops import ArrayOps


@dataclass(slots=True)
class LossInputs[A]:
    """Everything a strategy might need to compute a loss.

    Fields beyond ``student_logits`` / ``labels`` / ``loss_mask`` are optional and
    populated only by backends that support them, so today's response
    distillation and tomorrow's logit/hidden/attention strategies share one
    container. A strategy declares its needs via its ``requires_*`` flags and the
    backend supplies exactly those tensors.

    Shapes (B=batch, T=seq, V=vocab, L=layers, H=hidden):
        student_logits: ``[B, T, V]``
        labels:         ``[B, T]`` (int) — next-token targets.
        loss_mask:      ``[B, T]`` — 1 on supervised (completion) tokens else 0.
        teacher_logits: ``[B, T, V]`` — for logit distillation.
        student_hidden / teacher_hidden: tuple of ``[B, T, H]`` per layer.
    """

    student_logits: A
    labels: A
    loss_mask: A
    teacher_logits: A | None = None
    student_hidden: tuple[A, ...] = field(default_factory=tuple)
    teacher_hidden: tuple[A, ...] = field(default_factory=tuple)
    student_attentions: tuple[A, ...] = field(default_factory=tuple)
    teacher_attentions: tuple[A, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class LossResult[A]:
    """A computed loss plus a breakdown of its component terms.

    Attributes:
        total: The scalar loss array to differentiate.
        components: Named scalar sub-loss *arrays* (same backend type as
            ``total``), e.g. ``{"ce": <array>, "kd": <array>}``. They are kept as
            arrays — not Python floats — so building them never forces
            materialization inside an autograd trace; the backend converts them
            with :meth:`ArrayOps.to_float` only *after* the backward pass.
    """

    total: A
    components: dict[str, A] = field(default_factory=dict)


class DistillationStrategy[A](ABC):
    """Base class for all distillation objectives.

    Subclasses declare which optional teacher signals they consume so the trainer
    can validate configuration up front (e.g. refuse logit distillation when the
    teacher provides no logits) and skip computing signals nobody needs.
    """

    #: Whether ``LossInputs.teacher_logits`` must be provided.
    requires_teacher_logits: bool = False
    #: Whether hidden states must be provided.
    requires_hidden_states: bool = False
    #: Whether attention maps must be provided.
    requires_attentions: bool = False

    def __init__(self, config: DistillationConfig) -> None:
        """Store the distillation config (uniform constructor for every strategy)."""
        self._config = config

    @property
    @abstractmethod
    def kind(self) -> DistillationStrategyKind:
        """The strategy kind this implements."""

    @abstractmethod
    def compute_loss(self, inputs: LossInputs[A], ops: ArrayOps[A]) -> LossResult[A]:
        """Compute the (differentiable) distillation loss.

        Args:
            inputs: The tensors produced by a forward pass.
            ops: Backend array algebra (numpy in tests, MLX/PyTorch in training).

        Returns:
            A :class:`LossResult` whose ``total`` is differentiable w.r.t. the
            student parameters under the active backend's autograd.
        """

    def validate_inputs(self, inputs: LossInputs[A]) -> None:
        """Raise if the strategy's required signals are missing.

        Called by the trainer before the first step to fail fast on
        misconfiguration rather than deep inside the training loop.
        """
        if self.requires_teacher_logits and inputs.teacher_logits is None:
            raise ValueError(f"{type(self).__name__} requires teacher_logits.")
        if self.requires_hidden_states and not inputs.teacher_hidden:
            raise ValueError(f"{type(self).__name__} requires hidden states.")
        if self.requires_attentions and not inputs.teacher_attentions:
            raise ValueError(f"{type(self).__name__} requires attentions.")


# ---------------------------------------------------------------------------
# Strategy registry / factory.
# ---------------------------------------------------------------------------
_STRATEGIES: dict[DistillationStrategyKind, type[DistillationStrategy[Any]]] = {}


def register_strategy(
    kind: DistillationStrategyKind,
) -> Callable[[type[DistillationStrategy[Any]]], type[DistillationStrategy[Any]]]:
    """Class decorator registering a strategy under its kind.

    Example::

        @register_strategy(DistillationStrategyKind.RESPONSE)
        class ResponseDistillation(DistillationStrategy): ...
    """

    def _decorator(cls: type[DistillationStrategy[Any]]) -> type[DistillationStrategy[Any]]:
        _STRATEGIES[kind] = cls
        return cls

    return _decorator


def get_strategy(config: DistillationConfig) -> DistillationStrategy[Any]:
    """Instantiate the configured distillation strategy.

    Args:
        config: The distillation section of an experiment config.

    Returns:
        A ready :class:`DistillationStrategy`.

    Raises:
        NotImplementedError: If the requested strategy is declared but not yet
            implemented (e.g. logit/hidden/attention on the current roadmap).
    """
    # Import for side-effect registration; kept local to avoid import cycles.
    from distillation.losses import response  # noqa: F401

    strategy_cls = _STRATEGIES.get(config.strategy)
    if strategy_cls is None:
        raise NotImplementedError(
            f"Distillation strategy '{config.strategy.value}' is declared in the "
            "config schema but not yet implemented. Available: "
            f"{sorted(k.value for k in _STRATEGIES)}."
        )
    return strategy_cls(config)
