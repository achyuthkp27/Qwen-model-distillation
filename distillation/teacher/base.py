"""Abstract teacher-client interface.

Any teacher backend implements :class:`TeacherClient`. The generation pipeline
depends only on this abstraction (Dependency Inversion), so adding a second
teacher for multi-teacher distillation later requires no pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from distillation.datasets.schema import TokenUsage


@dataclass(frozen=True, slots=True)
class TeacherResponse:
    """Normalized result of a single teacher generation.

    Attributes:
        text: The generated completion (the final answer, not the reasoning).
        usage: Prompt/completion token counts (best-effort per backend).
        latency_s: Wall-clock latency measured by the client.
        thinking: The reasoning stream for reasoning-capable models, if any
            (captured for provenance; not used as the training target).
    """

    text: str
    usage: TokenUsage
    latency_s: float
    thinking: str = ""

    @property
    def tokens_per_sec(self) -> float:
        """Completion tokens per second, or 0 when unmeasurable."""
        if self.latency_s <= 0 or self.usage.completion_tokens <= 0:
            return 0.0
        return self.usage.completion_tokens / self.latency_s


class TeacherClient(ABC):
    """Interface every teacher backend must implement."""

    @property
    @abstractmethod
    def model(self) -> str:
        """The concrete model identifier this client generates from."""

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> TeacherResponse:
        """Generate a completion for a single prompt.

        Args:
            prompt: The user prompt.
            system: Optional system prompt overriding any client default.

        Returns:
            A :class:`TeacherResponse`.

        Raises:
            TeacherError: On unrecoverable backend failure.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` if the backend is reachable and the model available."""

    def close(self) -> None:  # noqa: B027 - intentional optional no-op hook, not abstract.
        """Release any held resources (connections, sessions). Optional."""


class TeacherError(RuntimeError):
    """Raised when a teacher backend fails irrecoverably."""
