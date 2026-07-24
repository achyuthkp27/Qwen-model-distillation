"""Backend-agnostic array operations for distillation losses.

The whole point of the loss layer is that a strategy's *math* is written once and
runs on any backend. We achieve that by expressing every strategy in terms of a
small :class:`ArrayOps` protocol rather than concrete MLX/PyTorch calls.

* :class:`NumpyOps` — a pure-NumPy implementation used for unit-testing loss math
  on CPU without any ML framework installed.
* MLX and PyTorch backends provide their own ``ArrayOps`` (with autograd) so the
  identical strategy code differentiates correctly during training.

Only the handful of primitives needed by the current + planned strategies
(cross-entropy, KL divergence) are exposed, keeping new backends cheap to add.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Arrays are backend tensors; strategies stay generic over this type var.
A = TypeVar("A")


@runtime_checkable
class ArrayOps(Protocol[A]):
    """Minimal array algebra a distillation strategy needs.

    All operations act on the last axis where a vocabulary axis is implied
    (``log_softmax``, ``softmax``, ``gather_last``, ``sum_last``).
    """

    def log_softmax(self, x: A) -> A:
        """Numerically stable log-softmax over the last axis."""
        ...

    def softmax(self, x: A) -> A:
        """Softmax over the last axis."""
        ...

    def gather_last(self, x: A, idx: A) -> A:
        """Gather values from ``x`` (``[..., V]``) at ``idx`` (``[...]``)."""
        ...

    def masked_mean(self, x: A, mask: A) -> A:
        """Mean of ``x`` over positions where ``mask`` != 0 (scalar array)."""
        ...

    def sum_last(self, x: A) -> A:
        """Sum over the last axis."""
        ...

    def scalar_mul(self, x: A, s: float) -> A:
        """Multiply an array by a Python float."""
        ...

    def add(self, a: A, b: A) -> A:
        """Element-wise addition."""
        ...

    def sub(self, a: A, b: A) -> A:
        """Element-wise subtraction."""
        ...

    def mul(self, a: A, b: A) -> A:
        """Element-wise multiplication."""
        ...

    def neg(self, x: A) -> A:
        """Element-wise negation."""
        ...

    def to_float(self, x: A) -> float:
        """Materialize a scalar array as a Python float (for logging)."""
        ...


class NumpyOps:
    """Pure-NumPy :class:`ArrayOps`, used for testing strategy math on CPU."""

    def log_softmax(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Numerically stable log-softmax over the last axis."""
        shifted = x - np.max(x, axis=-1, keepdims=True)
        return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

    def softmax(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Softmax over the last axis."""
        shifted = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return shifted / np.sum(shifted, axis=-1, keepdims=True)

    def gather_last(self, x: NDArray[np.float64], idx: NDArray[np.int64]) -> NDArray[np.float64]:
        """Gather along the last axis using integer indices."""
        return np.take_along_axis(x, idx[..., None], axis=-1)[..., 0]

    def masked_mean(self, x: NDArray[np.float64], mask: NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean over masked (non-zero) positions; safe against empty masks."""
        total = np.sum(x * mask)
        count = np.maximum(np.sum(mask), 1.0)
        return np.asarray(total / count, dtype=np.float64)

    def sum_last(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sum over the last axis."""
        return np.sum(x, axis=-1)

    def scalar_mul(self, x: NDArray[np.float64], s: float) -> NDArray[np.float64]:
        """Multiply by a scalar."""
        return x * s

    def add(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        """Element-wise addition."""
        return a + b

    def sub(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        """Element-wise subtraction."""
        return a - b

    def mul(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        """Element-wise multiplication."""
        return a * b

    def neg(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Element-wise negation."""
        return -x

    def to_float(self, x: NDArray[np.float64]) -> float:
        """Materialize a scalar as a float."""
        return float(x)
