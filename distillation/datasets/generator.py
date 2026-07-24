"""Phase 1 dataset-generation pipeline.

Reads a JSONL of prompts, queries the teacher for each, and streams typed
:class:`GenerationRecord` rows to disk. Designed for a long-running, potentially
interrupted job on a single machine:

* **Resumable** — already-generated prompt ids are detected and skipped, so a
  crashed run can be re-invoked to fill only the gaps.
* **Crash-resilient** — records are appended one line at a time (JSONL), so
  partial progress is never lost.
* **Backend-agnostic** — depends only on the :class:`TeacherClient` interface.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from distillation.datasets.schema import GenerationRecord, PromptRecord
from distillation.utils.io import read_jsonl, write_jsonl
from distillation.utils.logging import get_logger

if TYPE_CHECKING:
    # Imported for typing only; deferred at runtime to avoid a teacher<->datasets
    # import cycle (teacher.base depends on datasets.schema).
    from distillation.teacher.base import TeacherClient

_logger = get_logger(__name__)


@dataclass(slots=True)
class GenerationSummary:
    """Aggregate outcome of a generation run.

    Attributes:
        total_prompts: Prompts present in the input file.
        generated: Successful generations produced this run.
        skipped: Prompts skipped because they already existed (resume).
        failed: Prompts that errored (an error record is still written).
        output_path: File the records were written to.
    """

    total_prompts: int = 0
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    output_path: Path | None = None
    _completion_tokens: int = field(default=0, repr=False)
    _latency_s: float = field(default=0.0, repr=False)

    @property
    def mean_tokens_per_sec(self) -> float:
        """Mean teacher throughput across successful generations."""
        if self._latency_s <= 0:
            return 0.0
        return self._completion_tokens / self._latency_s


class DatasetGenerator:
    """Orchestrates teacher generation over a prompt file (Phase 1).

    Args:
        teacher: Any :class:`TeacherClient` implementation.
    """

    def __init__(self, teacher: TeacherClient) -> None:
        self._teacher = teacher  # any TeacherClient (imported lazily; see module top)

    @staticmethod
    def _load_prompts(prompts_path: Path) -> list[PromptRecord]:
        """Parse and validate the input prompt file."""
        prompts = [PromptRecord.model_validate(row) for row in read_jsonl(prompts_path)]
        ids = [p.id for p in prompts]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate prompt ids found; ids must be unique.")
        return prompts

    @staticmethod
    def _existing_ids(output_path: Path) -> set[int | str]:
        """Return ids of successful generations already on disk (for resume)."""
        if not output_path.is_file():
            return set()
        done: set[int | str] = set()
        for row in read_jsonl(output_path):
            # Only treat successful, non-empty generations as "done".
            if row.get("error") is None and row.get("response"):
                done.add(row["id"])
        return done

    def _generate_one(self, prompt: PromptRecord) -> GenerationRecord:
        """Generate a single record, capturing errors as data rather than raising."""
        # Local import breaks the teacher<->datasets module cycle at load time.
        from distillation.teacher.base import TeacherError

        try:
            result = self._teacher.generate(prompt.prompt, system=prompt.system)
            # A reasoning model can exhaust its token budget on the <think> stream
            # and return empty content; surface that as an actionable error rather
            # than a silent empty sample.
            error = None
            if not result.text.strip():
                error = (
                    "Empty response. If the teacher is a reasoning model, it likely "
                    "spent its token budget thinking; set generation.think=false or "
                    "raise generation.max_tokens."
                )
                _logger.warning("Empty response for prompt id=%s (%s)", prompt.id, error)
            return GenerationRecord(
                id=prompt.id,
                prompt=prompt.prompt,
                system=prompt.system,
                response=result.text,
                teacher_model=self._teacher.model,
                usage=result.usage,
                latency_s=result.latency_s,
                tokens_per_sec=result.tokens_per_sec,
                thinking=result.thinking or None,
                error=error,
            )
        except TeacherError as exc:
            _logger.error("Generation failed for prompt id=%s: %s", prompt.id, exc)
            return GenerationRecord(
                id=prompt.id,
                prompt=prompt.prompt,
                system=prompt.system,
                response="",
                teacher_model=self._teacher.model,
                latency_s=0.0,
                error=str(exc),
            )

    def _record_outcome(
        self, record: GenerationRecord, output_path: Path, summary: GenerationSummary
    ) -> None:
        """Persist one record and fold it into the running summary.

        Called only from the main thread so JSONL appends and summary mutation
        need no locking even when generation itself runs concurrently.
        """
        # Append immediately so a crash keeps everything generated so far.
        write_jsonl(output_path, [record.model_dump()], append=True)
        if record.ok:
            summary.generated += 1
            summary._completion_tokens += record.usage.completion_tokens
            summary._latency_s += record.latency_s
        else:
            summary.failed += 1

    def run(
        self,
        prompts_path: Path | str,
        output_path: Path | str,
        *,
        resume: bool = True,
        limit: int | None = None,
        max_concurrency: int = 1,
    ) -> GenerationSummary:
        """Generate teacher responses for every prompt.

        Args:
            prompts_path: Input JSONL of ``{id, prompt[, system]}`` rows.
            output_path: Destination JSONL for :class:`GenerationRecord` rows.
            resume: Skip prompts already successfully generated in the output.
            limit: If set, restrict the run to the **first ``limit`` prompts** of
                the file (a *target total*), not a per-run budget. Combined with
                ``resume`` this is idempotent: re-running with the same ``limit``
                converges to exactly ``limit`` generated rows and never exceeds
                it, regardless of how many prior partial runs occurred.
            max_concurrency: Number of teacher requests to run in parallel. >1
                uses a thread pool (teacher calls are IO-bound). Requires the
                Ollama server to allow parallelism (``OLLAMA_NUM_PARALLEL``).
                Generation runs on worker threads; all writes/summary updates
                happen on the main thread, so the JSONL stays consistent.

        Returns:
            A :class:`GenerationSummary` describing the run.
        """
        prompts_path = Path(prompts_path)
        output_path = Path(output_path)

        prompts = self._load_prompts(prompts_path)
        # Apply the limit to the target *set* first, then subtract what's already
        # done — so limit means "first N total", making resume idempotent.
        target = prompts[:limit] if limit is not None else prompts
        done_ids = self._existing_ids(output_path) if resume else set()
        pending = [p for p in target if p.id not in done_ids]

        summary = GenerationSummary(
            total_prompts=len(prompts),
            skipped=len(target) - len(pending),
            output_path=output_path,
        )
        _logger.info(
            "Generating with teacher=%s | prompts=%d pending=%d skipped=%d concurrency=%d -> %s",
            self._teacher.model,
            len(prompts),
            len(pending),
            summary.skipped,
            max_concurrency,
            output_path,
        )

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task("teacher generation", total=len(pending))

            if max_concurrency <= 1 or len(pending) <= 1:
                # Sequential path.
                for prompt in pending:
                    self._record_outcome(self._generate_one(prompt), output_path, summary)
                    progress.advance(task)
            else:
                # Concurrent path: fan generation out to worker threads, but
                # consume results (write + summary + progress) on this thread.
                with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
                    futures = {pool.submit(self._generate_one, prompt) for prompt in pending}
                    for future in as_completed(futures):
                        self._record_outcome(future.result(), output_path, summary)
                        progress.advance(task)

        _logger.info(
            "Done: generated=%d failed=%d skipped=%d | mean %.1f tok/s",
            summary.generated,
            summary.failed,
            summary.skipped,
            summary.mean_tokens_per_sec,
        )
        return summary
