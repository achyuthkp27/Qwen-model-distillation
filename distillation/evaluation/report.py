"""HTML evaluation report generation (Phase 5).

Renders an :class:`EvalResults` into a self-contained, dependency-free HTML page
comparing every subject side by side, and optionally embeds the Phase 8 training
plots. Kept as plain string templating (no Jinja) so it has zero runtime deps.
"""

from __future__ import annotations

import base64
import html
from datetime import UTC, datetime
from pathlib import Path

from distillation.evaluation.evaluator import EvalResults, SubjectResult
from distillation.utils.logging import get_logger

_logger = get_logger(__name__)

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem auto; max-width: 1000px; color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.6rem; } h2 { margin-top: 2rem; color: #333; }
table { border-collapse: collapse; width: 100%; background: #fff; margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); }
th, td { padding: .6rem .8rem; text-align: left; border-bottom: 1px solid #eee; }
th { background: #f2f4f7; font-weight: 600; }
tr:hover td { background: #f9fbff; }
.metric { font-variant-numeric: tabular-nums; }
.best { color: #0a7d28; font-weight: 600; }
img { max-width: 100%; border: 1px solid #eee; border-radius: 6px; margin: .5rem 0; }
.muted { color: #888; font-size: .85rem; }
code { background: #f0f0f0; padding: .1rem .3rem; border-radius: 3px; }
"""


def _fmt(value: float) -> str:
    """Format a float compactly for the report."""
    return f"{value:.3f}"


def _perf_table(subjects: list[SubjectResult]) -> str:
    """Render the performance comparison table."""
    rows = []
    # Higher tokens/sec is better; lower latency is better.
    best_tps = max((s.mean_tokens_per_sec for s in subjects), default=0.0)
    best_lat = min((s.mean_latency_s for s in subjects if s.mean_latency_s > 0), default=0.0)
    for s in subjects:
        tps_cls = " class='best'" if s.mean_tokens_per_sec == best_tps and best_tps else ""
        lat_cls = " class='best'" if s.mean_latency_s == best_lat and best_lat else ""
        rows.append(
            f"<tr><td>{html.escape(s.name)}</td>"
            f"<td class='metric'{lat_cls}>{_fmt(s.mean_latency_s)}</td>"
            f"<td class='metric'{tps_cls}>{_fmt(s.mean_tokens_per_sec)}</td>"
            f"<td class='metric'>{_fmt(s.peak_memory_mb)}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Subject</th><th>Mean latency (s)</th>"
        "<th>Tokens/sec</th><th>Peak memory (MiB)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _quality_table(subjects: list[SubjectResult]) -> str:
    """Render the text-similarity table (students vs teacher reference)."""
    metric_names = sorted({k for s in subjects for k in s.text_metrics})
    if not metric_names:
        return "<p class='muted'>No text metrics computed (install <code>.[eval]</code>).</p>"
    header = "".join(f"<th>{html.escape(m)}</th>" for m in metric_names)
    rows = []
    for s in subjects:
        if not s.text_metrics:
            continue
        cells = "".join(
            f"<td class='metric'>{_fmt(s.text_metrics.get(m, 0.0))}</td>" for m in metric_names
        )
        rows.append(f"<tr><td>{html.escape(s.name)}</td>{cells}</tr>")
    return (
        f"<table><thead><tr><th>Subject</th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _embed_plots(plot_paths: list[Path]) -> str:
    """Base64-embed PNG plots so the report is a single portable file."""
    blocks = []
    for path in plot_paths:
        if not path.is_file():
            continue
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        blocks.append(
            f"<figure><figcaption class='muted'>{html.escape(path.stem)}</figcaption>"
            f"<img src='data:image/png;base64,{data}' alt='{html.escape(path.stem)}'></figure>"
        )
    return "".join(blocks)


def render_report(
    results: EvalResults,
    output_path: Path,
    *,
    experiment_name: str = "experiment",
    plot_paths: list[Path] | None = None,
) -> Path:
    """Render an :class:`EvalResults` to a self-contained HTML file.

    Args:
        results: The evaluation results to render.
        output_path: Destination ``.html`` file.
        experiment_name: Shown in the report title.
        plot_paths: Optional training plots (PNGs) to embed.

    Returns:
        The path written.
    """
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    plots_html = _embed_plots(plot_paths or [])
    plots_section = f"<h2>Training plots</h2>{plots_html}" if plots_html else ""

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Distillation report — {html.escape(experiment_name)}</title>
<style>{_CSS}</style></head><body>
<h1>Distillation evaluation — {html.escape(experiment_name)}</h1>
<p class="muted">Generated {generated} · {len(results.prompts)} prompts · reference = teacher</p>
<h2>Quality (vs teacher)</h2>
{_quality_table(results.subjects)}
<h2>Performance</h2>
{_perf_table(results.subjects)}
{plots_section}
</body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    _logger.info("Wrote HTML report -> %s", output_path)
    return output_path
