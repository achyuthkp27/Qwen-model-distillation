"""Interactive chat REPL (Phase 6).

A dependency-light terminal loop over an :class:`InferenceEngine`. Kept separate
from the CLI so it can be driven programmatically (and tested) with injected
input/output callables.
"""

from __future__ import annotations

from collections.abc import Callable

from distillation.inference.engine import InferenceEngine

_EXIT_COMMANDS = frozenset({"/exit", "/quit", "/q"})


def run_chat(
    engine: InferenceEngine,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run an interactive chat loop until the user exits.

    Args:
        engine: The inference engine to converse with.
        input_fn: Prompt-reading callable (injectable for tests).
        output_fn: Output callable (injectable for tests).
    """
    output_fn("Chat with the student. Type /exit to quit.")
    engine.load()
    while True:
        try:
            user = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            break
        if not user:
            continue
        if user.lower() in _EXIT_COMMANDS:
            break
        result = engine.generate(user)
        output_fn(f"student> {result.text}")
        output_fn(f"  ({result.num_tokens} tok, {result.tokens_per_sec:.1f} tok/s)")
