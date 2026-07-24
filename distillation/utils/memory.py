"""Process memory probing.

On Apple Silicon the GPU shares unified memory with the CPU, so resident set
size (RSS) is a meaningful proxy for total footprint during training and
inference. We keep this dependency-free (``resource`` + ``os``) so it works even
without the ML backends installed.
"""

from __future__ import annotations

import os
import resource
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """A point-in-time view of process memory usage.

    Attributes:
        rss_mb: Resident set size in mebibytes.
        peak_rss_mb: Peak resident set size in mebibytes since process start.
    """

    rss_mb: float
    peak_rss_mb: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation."""
        return {"rss_mb": self.rss_mb, "peak_rss_mb": self.peak_rss_mb}


def _peak_rss_mb() -> float:
    """Return peak RSS in MiB via ``getrusage``.

    ``ru_maxrss`` is reported in *bytes* on macOS/Darwin and in *kibibytes* on
    Linux, so we normalize based on platform.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 * 1024)  # bytes -> MiB
    return peak / 1024  # KiB -> MiB


def _current_rss_mb() -> float:
    """Return current RSS in MiB.

    Reads ``/proc`` on Linux; on macOS there is no cheap portable current-RSS
    read without extra deps, so we fall back to the peak value.
    """
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{os.getpid()}/statm", encoding="ascii") as fh:
                pages = int(fh.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        except (OSError, ValueError, IndexError):
            return _peak_rss_mb()
    return _peak_rss_mb()


def current_memory() -> MemorySnapshot:
    """Capture the current process memory footprint."""
    return MemorySnapshot(rss_mb=_current_rss_mb(), peak_rss_mb=_peak_rss_mb())
