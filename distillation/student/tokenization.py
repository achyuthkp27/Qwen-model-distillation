"""Shared training-sample tokenization for student backends (Phase 3).

Both the MLX and PEFT backends turn a formatted sample (chat / completion /
instruction) into the same ``input_ids / labels / loss_mask`` triple, masking the
prompt so only the teacher completion is supervised. That logic lives here once
(DRY) and is version-robust:

`apply_chat_template(tokenize=True)` changed return type across Transformers
versions (a plain list in 4.x, a ``BatchEncoding`` in 5.x). Slicing the latter
silently yields empty ids. To avoid that, we always render the template to a
*string* and tokenize the string, which is stable everywhere.
"""

from __future__ import annotations

from typing import Any, Protocol

from distillation.student.base import EncodedSample


class _Tokenizer(Protocol):
    """The minimal tokenizer surface this module needs."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """Tokenize a string into ids."""
        ...

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        """Render chat messages via the model's chat template."""
        ...


def _ids_from_messages(
    tok: _Tokenizer,
    messages: list[dict[str, str]],
    enable_thinking: bool | None = None,
) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, full_ids) for a chat sample.

    The prompt is everything up to (and including) the generation prompt; the
    full sequence additionally contains the assistant answer.

    ``enable_thinking`` (when not ``None``) is forwarded to the chat template for
    reasoning models (Qwen3): passing ``False`` keeps training consistent with
    think-free generation so the prompt/answer boundary stays aligned.
    """
    extra: dict[str, Any] = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    *context, _answer = messages
    prompt_text = tok.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False, **extra
    )
    full_text = tok.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False, **extra
    )
    # The template already emits special tokens as text; don't double them.
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    return prompt_ids, full_ids


def encode_sample(
    tok: _Tokenizer,
    sample: dict[str, Any],
    max_seq_length: int,
    enable_thinking: bool | None = None,
) -> EncodedSample:
    """Tokenize one formatted sample into ids/labels/mask.

    Supports all three Phase-2 formats:
      * chat        -> ``{"messages": [...]}``
      * completion  -> ``{"prompt": ..., "completion": ...}``
      * instruction -> ``{"instruction": ..., "output": ...}``

    Args:
        tok: The model tokenizer (HF or mlx-lm wrapper).
        sample: One formatted training row.
        max_seq_length: Truncation length for the full sequence.

    Returns:
        An :class:`EncodedSample` with the prompt span masked out of the loss.
    """
    if "messages" in sample:
        prompt_ids, full_ids = _ids_from_messages(tok, sample["messages"], enable_thinking)
    else:
        prompt_text = sample.get("prompt", sample.get("instruction", ""))
        completion = sample.get("completion", sample.get("output", ""))
        prompt_ids = tok.encode(prompt_text)
        full_ids = tok.encode(prompt_text + completion)

    full_ids = full_ids[:max_seq_length]
    # Guard against degenerate rows (empty completion) producing a 0-length input.
    if len(full_ids) < 2:
        full_ids = (full_ids + [0, 0])[:2]

    n_prompt = min(len(prompt_ids), len(full_ids))
    # Next-token targets; supervise only the completion tokens.
    input_ids = full_ids[:-1]
    labels = full_ids[1:]
    loss_mask = [0] * max(n_prompt - 1, 0) + [1] * (len(input_ids) - max(n_prompt - 1, 0))
    loss_mask = loss_mask[: len(input_ids)]
    return EncodedSample(input_ids=input_ids, labels=labels, loss_mask=loss_mask)
