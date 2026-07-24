"""Side-by-side teacher vs student comparison (Phase 6 helper).

Lets you ask both models the same question and see their answers together — the
most direct way to *feel* how well distillation worked. The core
:func:`compare_once` is backend-agnostic and pure (easy to test); the CLI adds
the pretty rendering and the interactive loop on top.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from distillation.inference.engine import InferenceEngine
from distillation.teacher.base import TeacherClient


@dataclass(slots=True)
class Comparison:
    """Both models' answers to one prompt, with timing.

    Attributes:
        prompt: The question asked.
        teacher_text / student_text: The two answers.
        teacher_latency_s / student_latency_s: Wall-clock time each took.
        teacher_tokens / student_tokens: Completion token counts.
    """

    prompt: str
    teacher_text: str
    teacher_latency_s: float
    teacher_tokens: int
    student_text: str
    student_latency_s: float
    student_tokens: int

    @property
    def speedup(self) -> float:
        """How many times faster the student answered (0 if unmeasurable)."""
        if self.student_latency_s <= 0:
            return 0.0
        return self.teacher_latency_s / self.student_latency_s


def compare_once(teacher: TeacherClient, engine: InferenceEngine, prompt: str) -> Comparison:
    """Ask the teacher and the student the same prompt and collect both answers.

    Args:
        teacher: The teacher client (honours the config's ``think`` setting).
        engine: The student inference engine (loads its adapter on first use).
        prompt: The question to ask both.

    Returns:
        A :class:`Comparison` with both answers and their timings.
    """
    teacher_resp = teacher.generate(prompt)
    student_resp = engine.generate(prompt)
    return Comparison(
        prompt=prompt,
        teacher_text=teacher_resp.text,
        teacher_latency_s=teacher_resp.latency_s,
        teacher_tokens=teacher_resp.usage.completion_tokens,
        student_text=student_resp.text,
        student_latency_s=student_resp.latency_s,
        student_tokens=student_resp.num_tokens,
    )


def run_compare_loop(
    teacher: TeacherClient,
    engine: InferenceEngine,
    render: Callable[[Comparison], None],
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Interactive loop: ask a question, render both answers, repeat.

    Args:
        teacher: Teacher client.
        engine: Student engine.
        render: Callback that displays a :class:`Comparison` (the CLI passes a
            Rich renderer; tests pass a simple collector).
        input_fn: Reads the next question (injectable for tests).
        output_fn: Prints prompts/status (injectable for tests).
    """
    output_fn("Ask a question to compare teacher vs student. /exit to quit.")
    engine.load()
    while True:
        try:
            question = input_fn("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            break
        if not question:
            continue
        if question.lower() in {"/exit", "/quit", "/q"}:
            break
        render(compare_once(teacher, engine, question))
