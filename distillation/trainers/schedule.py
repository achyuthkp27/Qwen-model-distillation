"""Learning-rate schedules.

A schedule maps a global optimizer step to a learning rate. Kept tiny and pure
(no framework deps) so it is trivially testable and reusable across backends.
"""

from __future__ import annotations

from collections.abc import Callable

LRSchedule = Callable[[int], float]


def warmup_constant(base_lr: float, warmup_steps: int) -> LRSchedule:
    """Linear warmup from 0 to ``base_lr`` over ``warmup_steps``, then constant.

    Args:
        base_lr: The target (post-warmup) learning rate.
        warmup_steps: Number of steps to ramp up over (0 disables warmup).

    Returns:
        A callable ``step -> lr`` (``step`` is 1-indexed).
    """

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * step / warmup_steps
        return base_lr

    return schedule
