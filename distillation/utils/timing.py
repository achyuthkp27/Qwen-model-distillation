"""Lightweight wall-clock timing helpers.

Latency measurement is a first-class metric for distillation (Phase 5), so we
expose a reusable context manager / manual stopwatch rather than sprinkling
``time.perf_counter()`` calls throughout the codebase.
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Self


class Stopwatch:
    """Measure elapsed wall-clock time.

    Usable either as a context manager::

        with Stopwatch() as sw:
            do_work()
        print(sw.elapsed_s)

    or manually via :meth:`start` / :meth:`stop`.
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._elapsed_s: float = 0.0

    def start(self) -> Self:
        """Begin timing and return self for chaining."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop timing and return the elapsed seconds."""
        if self._start is None:
            raise RuntimeError("Stopwatch.stop() called before start().")
        self._elapsed_s = time.perf_counter() - self._start
        self._start = None
        return self._elapsed_s

    @property
    def elapsed_s(self) -> float:
        """Elapsed seconds from the most recent completed interval."""
        return self._elapsed_s

    @property
    def elapsed_ms(self) -> float:
        """Elapsed milliseconds from the most recent completed interval."""
        return self._elapsed_s * 1000.0

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
