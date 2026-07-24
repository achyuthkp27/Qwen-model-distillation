"""Tests for the shared student tokenization (prompt masking, robustness).

Uses a fake tokenizer so no ML framework is required. This locks in the fix for
the Transformers-5.x ``apply_chat_template(tokenize=True)`` return-type change by
exercising the string-render path the real backends now use.
"""

from __future__ import annotations

from typing import Any

from distillation.student.tokenization import encode_sample


class _FakeTokenizer:
    """One token per whitespace word; chat template joins message contents."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool, tokenize: bool
    ) -> str:
        assert tokenize is False  # backends must request the string form
        parts = [m["content"] for m in messages]
        if add_generation_prompt:
            parts.append("<gen>")
        return " ".join(parts)

    def encode(self, text: str, **_: Any) -> list[int]:
        return [len(w) for w in text.split()] if text.strip() else []


def test_chat_sample_masks_prompt_tokens() -> None:
    tok = _FakeTokenizer()
    sample = {
        "messages": [
            {"role": "user", "content": "a bb"},
            {"role": "assistant", "content": "c dd e"},
        ]
    }
    enc = encode_sample(tok, sample, max_seq_length=128)

    # full = "a bb c dd e" -> 5 tokens; prompt = "a bb <gen>" -> 3 tokens.
    assert len(enc.input_ids) == 4  # full[:-1]
    assert len(enc.labels) == 4
    # First (n_prompt-1)=2 positions are prompt (masked), rest supervised.
    assert enc.loss_mask == [0, 0, 1, 1]


def test_completion_sample_tokenizes_prompt_plus_completion() -> None:
    tok = _FakeTokenizer()
    # Completions carry a leading separator in practice; the fake tokenizer is
    # whitespace-based, so include it to mirror real token boundaries.
    enc = encode_sample(tok, {"prompt": "hello world", "completion": " foo bar"}, 128)
    assert len(enc.input_ids) >= 1
    assert sum(enc.loss_mask) >= 1


def test_truncation_respects_max_seq_length() -> None:
    tok = _FakeTokenizer()
    sample = {
        "messages": [
            {"role": "user", "content": "w " * 50},
            {"role": "assistant", "content": "x " * 50},
        ]
    }
    enc = encode_sample(tok, sample, max_seq_length=10)
    assert len(enc.input_ids) == 9  # 10 truncated, minus the shift


def test_degenerate_empty_sample_does_not_crash() -> None:
    tok = _FakeTokenizer()
    enc = encode_sample(tok, {"prompt": "", "completion": ""}, 128)
    # Guard pads to a minimal 2-token sequence so attention never sees length 0.
    assert len(enc.input_ids) >= 1
