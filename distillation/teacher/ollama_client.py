"""Ollama-backed teacher client (Phase 1).

Talks to a local Ollama server via its HTTP API using the ``/api/chat`` endpoint
(non-streaming), which returns token counts we use for metrics. Network calls are
wrapped with bounded exponential-backoff retries for transient failures.
"""

from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from distillation.config.models import GenerationConfig, TeacherConfig
from distillation.datasets.schema import TokenUsage
from distillation.teacher.base import TeacherClient, TeacherError, TeacherResponse
from distillation.utils.logging import get_logger
from distillation.utils.timing import Stopwatch

_logger = get_logger(__name__)

# Errors worth retrying: transient network / 5xx conditions.
_RETRYABLE = (httpx.TransportError, httpx.HTTPStatusError)


class OllamaTeacherClient(TeacherClient):
    """Generate teacher completions from a local Ollama server.

    Args:
        config: Teacher configuration (model tag, connection, sampling).
        client: Optional pre-built ``httpx.Client`` (primarily for testing).
    """

    def __init__(self, config: TeacherConfig, *, client: httpx.Client | None = None) -> None:
        self._config = config
        self._model = config.model
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=config.ollama.base_url,
            timeout=config.ollama.request_timeout_s,
        )
        # Build a Tenacity-wrapped generate bound to the configured retry budget.
        self._generate_with_retry = retry(
            reraise=True,
            stop=stop_after_attempt(config.ollama.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(_RETRYABLE),
        )(self._generate_once)

    @property
    def model(self) -> str:
        """The Ollama model tag used for generation."""
        return self._model

    def _build_messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        """Assemble the chat messages, applying system-prompt precedence.

        Per-call ``system`` wins over the config-level default system prompt.
        """
        messages: list[dict[str, str]] = []
        effective_system = system or self._config.generation.system_prompt
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _sampling_options(gen: GenerationConfig) -> dict[str, object]:
        """Map our generation config to Ollama's ``options`` payload."""
        options: dict[str, object] = {
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "top_k": gen.top_k,
            "num_predict": gen.max_tokens,
        }
        if gen.seed is not None:
            options["seed"] = gen.seed
        return options

    def _generate_once(self, prompt: str, system: str | None) -> TeacherResponse:
        """Perform a single (retryable) generation request."""
        gen = self._config.generation
        payload: dict[str, object] = {
            "model": self._model,
            "messages": self._build_messages(prompt, system),
            "stream": False,
            "options": self._sampling_options(gen),
        }
        # Reasoning control is a top-level Ollama field, not a sampling option.
        # Only sent when explicitly configured so default model behavior is kept.
        if gen.think is not None:
            payload["think"] = gen.think

        with Stopwatch() as sw:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        message = data.get("message", {})
        text = message.get("content", "")
        # Reasoning models return their chain-of-thought in a separate field.
        thinking = message.get("thinking", "") or ""
        usage = TokenUsage(
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
        )
        return TeacherResponse(text=text, usage=usage, latency_s=sw.elapsed_s, thinking=thinking)

    def generate(self, prompt: str, *, system: str | None = None) -> TeacherResponse:
        """Generate a completion, retrying transient failures.

        Raises:
            TeacherError: If generation fails after exhausting retries.
        """
        try:
            return self._generate_with_retry(prompt, system)
        except _RETRYABLE as exc:
            raise TeacherError(
                f"Ollama generation failed for model {self._model!r}: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """Return ``True`` if the server is up and the model is pulled."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            tags = {m.get("name", "") for m in resp.json().get("models", [])}
        except _RETRYABLE as exc:
            _logger.warning("Ollama health check failed: %s", exc)
            return False
        # Ollama tags may or may not carry a ``:latest`` suffix; match loosely.
        available = any(t == self._model or t.startswith(f"{self._model}:") for t in tags)
        if not available:
            _logger.warning(
                "Model %r not found on Ollama. Available: %s. Run `ollama pull %s`.",
                self._model,
                sorted(tags),
                self._model,
            )
        return available

    def close(self) -> None:
        """Close the underlying HTTP client if we created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaTeacherClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
