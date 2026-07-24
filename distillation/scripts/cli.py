"""Unified ``distill`` command-line interface.

Sub-commands are added phase by phase. Phase 1 ships ``generate`` (teacher
dataset generation) and ``health`` (teacher connectivity check). Later phases
register ``format``, ``train``, ``evaluate``, ``infer``, and ``serve`` here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from distillation.config.loader import load_config
from distillation.datasets.builder import DatasetBuilder
from distillation.datasets.generator import DatasetGenerator
from distillation.datasets.schema import PromptRecord
from distillation.experiment import ExperimentManager
from distillation.inference.chat import run_chat
from distillation.inference.engine import InferenceEngine
from distillation.losses.base import get_strategy
from distillation.student.factory import get_backend
from distillation.teacher.ollama_client import OllamaTeacherClient
from distillation.trainers.metrics import MetricsRecorder
from distillation.trainers.trainer import DistillationTrainer
from distillation.utils.io import read_jsonl, write_jsonl
from distillation.utils.logging import get_logger, setup_logging
from distillation.utils.seed import seed_everything

app = typer.Typer(
    name="distill",
    help="Local LLM distillation framework (Apple Silicon).",
    no_args_is_help=True,
    add_completion=False,
)

_logger = get_logger(__name__)

_ConfigOpt = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="Experiment YAML config."),
]


@app.command()
def health(config: _ConfigOpt) -> None:
    """Check that the teacher (Ollama) is reachable and the model is pulled."""
    setup_logging()
    cfg = load_config(config)
    with OllamaTeacherClient(cfg.teacher) as teacher:
        ok = teacher.health_check()
    if ok:
        typer.secho(f"Teacher '{cfg.teacher.model}' is available.", fg=typer.colors.GREEN)
    else:
        typer.secho(
            f"Teacher '{cfg.teacher.model}' is NOT available. "
            f"Start Ollama and run `ollama pull {cfg.teacher.model}`.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


@app.command()
def generate(
    config: _ConfigOpt,
    resume: Annotated[
        bool, typer.Option(help="Skip prompts already generated in the output file.")
    ] = True,
    limit: Annotated[
        int | None,
        typer.Option(
            help="Generate the first N prompts total (idempotent with --resume; "
            "re-running with the same N never exceeds N)."
        ),
    ] = None,
) -> None:
    """Phase 1: generate teacher responses for every prompt in the dataset."""
    cfg = load_config(config)
    setup_logging(log_file=cfg.experiment_dir / "logs" / "generate.log")
    seed_everything(cfg.seed)

    with OllamaTeacherClient(cfg.teacher) as teacher:
        if not teacher.health_check():
            typer.secho(
                f"Teacher '{cfg.teacher.model}' unavailable; aborting.", fg=typer.colors.RED
            )
            raise typer.Exit(code=1)
        generator = DatasetGenerator(teacher)
        summary = generator.run(
            cfg.dataset.prompts_path,
            cfg.dataset.generations_path,
            resume=resume,
            limit=limit,
            max_concurrency=cfg.teacher.ollama.max_concurrency,
        )

    typer.secho(
        f"Generated {summary.generated} (failed {summary.failed}, skipped {summary.skipped}) "
        f"-> {summary.output_path} | mean {summary.mean_tokens_per_sec:.1f} tok/s",
        fg=typer.colors.GREEN,
    )


@app.command()
def format(config: _ConfigOpt) -> None:
    """Phase 2: convert teacher generations into a formatted train/val dataset."""
    cfg = load_config(config)
    setup_logging(log_file=cfg.experiment_dir / "logs" / "format.log")
    seed_everything(cfg.seed)

    if not cfg.dataset.generations_path.is_file():
        typer.secho(
            f"Generations file not found: {cfg.dataset.generations_path}. "
            "Run `distill generate` first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    summary = DatasetBuilder(cfg.dataset, seed=cfg.seed).build()
    typer.secho(
        f"Formatted '{cfg.dataset.format.value}': "
        f"{summary.train_size} train, {summary.val_size} val "
        f"({summary.skipped} skipped) -> {summary.train_path}",
        fg=typer.colors.GREEN,
    )


@app.command()
def train(config: _ConfigOpt) -> None:
    """Phase 3: LoRA-distill the student from the formatted dataset."""
    cfg = load_config(config)
    paths = ExperimentManager(cfg).setup()
    setup_logging(log_file=paths.logs / "train.log")
    seed_everything(cfg.seed)

    if not cfg.dataset.train_path.is_file():
        typer.secho(
            f"Train file not found: {cfg.dataset.train_path}. Run `distill format` first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    train_samples = list(read_jsonl(cfg.dataset.train_path))
    val_samples = (
        list(read_jsonl(cfg.dataset.val_path))
        if cfg.dataset.val_path and cfg.dataset.val_path.is_file()
        else None
    )

    # Assemble the backend (MLX/PEFT) and the distillation strategy from config.
    backend = get_backend(cfg.student)
    backend.load()
    # Train in the same think-free format the student generates in (reasoning models).
    backend.set_thinking(cfg.teacher.generation.think)
    backend.apply_lora(cfg.student.lora)
    backend.create_optimizer(cfg.training)
    strategy = get_strategy(cfg.distillation)

    recorder = MetricsRecorder(paths.metrics_file)
    trainer = DistillationTrainer(
        backend,
        strategy,
        cfg.training,
        recorder=recorder,
        adapter_dir=paths.adapters,
    )
    result = trainer.train(train_samples, val_samples)
    backend.close()

    typer.secho(
        f"Trained {result.total_steps} steps | final loss {result.final_train_loss:.4f} | "
        f"best val {result.best_val_loss} | adapter -> {result.adapter_path}",
        fg=typer.colors.GREEN,
    )


def _resolve_adapter(adapter_dir: Path) -> Path | None:
    """Return the final adapter path if a checkpoint exists, else ``None``."""
    for candidate in (adapter_dir / "adapter_final.safetensors", adapter_dir / "adapter_final"):
        if candidate.exists():
            return adapter_dir / "adapter_final"
    return None


@app.command()
def evaluate(
    config: _ConfigOpt,
    before: Annotated[
        bool, typer.Option(help="Include the untrained student for comparison.")
    ] = True,
) -> None:
    """Phase 5: compare teacher vs student (before/after) and write an HTML report."""
    from distillation.evaluation.evaluator import Evaluator
    from distillation.evaluation.metrics import get_text_metrics
    from distillation.evaluation.plots import generate_plots
    from distillation.evaluation.report import render_report
    from distillation.evaluation.sources import StudentSource, TeacherSource

    cfg = load_config(config)
    paths = ExperimentManager(cfg).setup()
    setup_logging(log_file=paths.logs / "evaluate.log")
    seed_everything(cfg.seed)

    # Prefer a held-out eval set (avoids measuring on training prompts); fall
    # back to the training prompt file when none is configured.
    eval_prompts_path = cfg.evaluation.eval_prompts_path or cfg.dataset.prompts_path
    prompts = [PromptRecord.model_validate(r).prompt for r in read_jsonl(eval_prompts_path)][
        : cfg.evaluation.num_samples
    ]
    _logger.info("Evaluating on %d prompts from %s", len(prompts), eval_prompts_path)

    metrics = get_text_metrics(
        cfg.evaluation.metrics, bertscore_model=cfg.evaluation.bertscore_model
    )
    teacher = TeacherSource(OllamaTeacherClient(cfg.teacher))

    students = []
    if before:
        b = get_backend(cfg.student)
        b.load()
        students.append(StudentSource(b, cfg.teacher.generation, "student_before"))

    adapter = _resolve_adapter(paths.adapters)
    if adapter is not None:
        a = get_backend(cfg.student)
        a.load()
        a.apply_lora(cfg.student.lora)
        a.load_adapter(adapter)
        students.append(StudentSource(a, cfg.teacher.generation, "student_after"))
    else:
        typer.secho(
            "No trained adapter found; evaluating teacher/before only.", fg=typer.colors.YELLOW
        )

    results = Evaluator(metrics).run(prompts, teacher, students)
    paths.eval_file.write_text(json.dumps(results.as_dict(), indent=2), encoding="utf-8")

    plots = generate_plots(paths.metrics_file, paths.plots)
    render_report(results, cfg.evaluation.report_path, experiment_name=cfg.name, plot_paths=plots)
    typer.secho(f"Report -> {cfg.evaluation.report_path}", fg=typer.colors.GREEN)


@app.command()
def plot(config: _ConfigOpt) -> None:
    """Phase 8: render training plots from the recorded metrics."""
    from distillation.evaluation.plots import generate_plots

    cfg = load_config(config)
    paths = ExperimentManager(cfg).setup()
    setup_logging()
    written = generate_plots(paths.metrics_file, paths.plots)
    typer.secho(f"Wrote {len(written)} plots -> {paths.plots}", fg=typer.colors.GREEN)


def _build_engine(config_path: Path) -> tuple[InferenceEngine, str]:
    """Load config and construct an inference engine with the trained adapter."""
    cfg = load_config(config_path)
    paths = ExperimentManager(cfg).setup()
    adapter = _resolve_adapter(paths.adapters)
    engine = InferenceEngine.from_config(cfg.student, cfg.teacher.generation, adapter_path=adapter)
    return engine, cfg.name


@app.command()
def infer(
    config: _ConfigOpt,
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Prompt to complete.")],
) -> None:
    """Phase 6: generate a single completion from the trained student."""
    setup_logging()
    engine, _ = _build_engine(config)
    result = engine.generate(prompt)
    engine.close()
    typer.echo(result.text)
    typer.secho(
        f"({result.num_tokens} tok, {result.tokens_per_sec:.1f} tok/s)", fg=typer.colors.CYAN
    )


@app.command()
def batch(
    config: _ConfigOpt,
    prompts_file: Annotated[Path, typer.Option(exists=True, help="JSONL of {id, prompt}.")],
    output: Annotated[Path, typer.Option(help="Destination JSONL for completions.")],
) -> None:
    """Phase 6: batch-generate completions for a prompt file."""
    setup_logging()
    engine, _ = _build_engine(config)
    prompts = [PromptRecord.model_validate(r) for r in read_jsonl(prompts_file)]
    results = engine.batch([p.prompt for p in prompts])
    engine.close()
    rows = [
        {"id": p.id, "prompt": p.prompt, "response": r.text, "num_tokens": r.num_tokens}
        for p, r in zip(prompts, results, strict=True)
    ]
    write_jsonl(output, rows)
    typer.secho(f"Wrote {len(rows)} completions -> {output}", fg=typer.colors.GREEN)


@app.command()
def chat(config: _ConfigOpt) -> None:
    """Phase 6: interactive chat with the trained student."""
    setup_logging()
    engine, _ = _build_engine(config)
    try:
        run_chat(engine)
    finally:
        engine.close()


@app.command()
def accuracy(
    config: _ConfigOpt,
    test_file: Annotated[
        Path, typer.Option(exists=True, help="Labelled JSONL: {id, prompt, label}.")
    ],
    subject: Annotated[
        str, typer.Option(help="Which to score: 'teacher', 'student', or 'both'.")
    ] = "both",
) -> None:
    """Measure classification accuracy vs gold labels (e.g. sentiment)."""
    from distillation.evaluation.classification import evaluate_accuracy
    from distillation.evaluation.sources import EngineSource, TeacherSource

    cfg = load_config(config)
    setup_logging()
    rows = list(read_jsonl(test_file))
    prompts = [r["prompt"] for r in rows]
    golds = [str(r["label"]).lower() for r in rows]
    typer.secho(f"Scoring {len(rows)} labelled examples from {test_file}", fg=typer.colors.CYAN)

    reports = []
    if subject in ("teacher", "both"):
        with OllamaTeacherClient(cfg.teacher) as teacher:
            reports.append(evaluate_accuracy(TeacherSource(teacher), prompts, golds))
    if subject in ("student", "both"):
        engine, _ = _build_engine(config)
        reports.append(evaluate_accuracy(EngineSource(engine, "student"), prompts, golds))
        engine.close()

    typer.echo("")
    for r in reports:
        typer.secho(
            f"{r.name:8s}  accuracy = {r.accuracy * 100:5.1f}%  "
            f"({r.correct}/{r.total}, {r.unparsed} unparsed)",
            fg=typer.colors.GREEN if r.accuracy >= 0.6 else typer.colors.YELLOW,
        )
        for lbl, (c, t) in r.per_class.items():
            typer.echo(f"           {lbl:9s}: {c}/{t} ({(c / t if t else 0) * 100:.0f}%)")


@app.command()
def compare(
    config: _ConfigOpt,
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="Ask once; omit for an interactive loop."),
    ] = None,
) -> None:
    """Phase 6: ask the teacher and student the same question, side by side."""
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel

    from distillation.inference.compare import Comparison, compare_once, run_compare_loop

    setup_logging()
    cfg = load_config(config)
    console = Console()

    teacher = OllamaTeacherClient(cfg.teacher)
    engine, _ = _build_engine(config)

    def render(c: Comparison) -> None:
        """Print both answers as side-by-side panels with timing."""
        teacher_panel = Panel(
            c.teacher_text or "[dim](empty)[/dim]",
            title=f"[bold]TEACHER[/bold] · {cfg.teacher.model}",
            subtitle=f"{c.teacher_latency_s:.2f}s · {c.teacher_tokens} tok",
            border_style="blue",
        )
        student_panel = Panel(
            c.student_text or "[dim](empty)[/dim]",
            title=f"[bold]STUDENT[/bold] · {cfg.student.model.split('/')[-1]}",
            subtitle=f"{c.student_latency_s:.2f}s · {c.student_tokens} tok",
            border_style="green",
        )
        console.print(f"\n[bold]Q:[/bold] {c.prompt}")
        console.print(Columns([teacher_panel, student_panel], equal=True, expand=True))
        if c.speedup:
            console.print(f"[dim]student was {c.speedup:.1f}x faster[/dim]")

    try:
        if prompt is not None:
            render(compare_once(teacher, engine, prompt))
        else:
            run_compare_loop(teacher, engine, render)
    finally:
        teacher.close()
        engine.close()


@app.command()
def serve(
    config: _ConfigOpt,
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Phase 6: serve the trained student over a REST API."""
    import uvicorn

    from distillation.inference.api import create_app

    setup_logging()
    engine, _ = _build_engine(config)
    uvicorn.run(create_app(engine), host=host, port=port)


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":
    main()
