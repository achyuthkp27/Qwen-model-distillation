# distillation
.venv/bin/distill compare --config configs/experiment_001.yaml -p "your question"

A local, production-quality **LLM distillation framework** optimized for **Apple
Silicon** (MLX + HuggingFace). It distills a large teacher into a small student
so the student imitates the teacher as closely as possible.

**Default pair:** teacher `qwen3:8b` → student `qwen3:4b` (fully configurable).

> Note: the framework is model-agnostic. `qwen3.5:9b/4b` do not exist as Ollama
> tags; `qwen3:8b → qwen3:4b` is the closest real, same-tokenizer-family pair,
> which is a prerequisite for future logit/KL distillation.

## Design principle

Distillation logic lives behind a single `DistillationStrategy` abstraction and
the student behind a `StudentBackend` interface, so **response distillation**
(shipping now) and future **logit / hidden-state / attention** objectives — and
either the **MLX** or **PEFT** backend — plug in *without touching the trainer*.

The loss math is written once against a tiny `ArrayOps` protocol, so the same
strategy code differentiates under NumPy (tests), MLX, and PyTorch.

## Architecture

```
distillation/
├── config/       Pydantic-validated config models + YAML loader        (Phase 9)
├── utils/        Logging, timing, memory probes, seeding, JSONL IO
├── teacher/      Teacher clients (Ollama) behind a TeacherClient iface  (Phase 1)
├── datasets/     Generation pipeline + format adapters                  (Phase 1-2)
├── losses/       DistillationStrategy ABC + ArrayOps + response loss    (Phase 4)
├── student/      MLX + PEFT backends behind one StudentBackend iface    (Phase 3)
├── trainers/     Training loop, LR schedule, metrics recording          (Phase 3)
├── evaluation/   BLEU/ROUGE/BERTScore + latency/mem + HTML + plots      (Phase 5,8)
├── inference/    Engine + REST API + chat REPL                          (Phase 6)
├── experiment.py Per-experiment directory management                    (Phase 7)
└── scripts/      Unified `distill` CLI
configs/          YAML experiment definitions
prompts/          Seed prompt JSONL files
experiments/      Per-experiment artifacts (config, logs, metrics, plots, adapters)
tests/            pytest suite (59 tests, no GPU/model required)
```

## Install

```bash
# Core + Apple-Silicon training backend + dev tooling
pip install -e ".[mlx,dev]"
# Optional extras: ".[peft]" (PyTorch/PEFT), ".[eval]" (metrics+plots), ".[serve]" (REST)
```

Requires Python 3.12+ and a running [Ollama](https://ollama.com) server.

## End-to-end workflow

```bash
ollama pull qwen3:8b                                   # teacher
distill health   --config configs/experiment_001.yaml  # check connectivity

distill generate --config configs/experiment_001.yaml  # Phase 1: teacher dataset
distill format   --config configs/experiment_001.yaml  # Phase 2: train/val split
distill train    --config configs/experiment_001.yaml  # Phase 3+4: LoRA distillation
distill evaluate --config configs/experiment_001.yaml  # Phase 5+8: report + plots
distill plot     --config configs/experiment_001.yaml  # Phase 8: plots only

distill infer  --config configs/experiment_001.yaml -p "Explain Redis."   # Phase 6
distill chat   --config configs/experiment_001.yaml                       # Phase 6
distill batch  --config configs/experiment_001.yaml \
               --prompts-file prompts/seed_prompts.jsonl --output out.jsonl
distill serve  --config configs/experiment_001.yaml --port 8000           # Phase 6 REST
```

### Reasoning-model teachers (e.g. `qwen3.5`)

Reasoning models emit a separate `<think>` stream and can spend their whole token
budget thinking, returning empty `content`. For distillation you want the final
answer, so set:

```yaml
teacher:
  generation:
    think: false   # disable the reasoning stream; return the answer directly
```

The teacher client captures any reasoning into each record's `thinking` field for
provenance, and empty responses are reported as failures with an actionable
message rather than silently producing blank training samples.

### Data formats (Phase 2)

`dataset.format` selects the training-sample shape:

| Format | Output shape |
|---|---|
| `instruction` | `{"instruction": ..., "input": "", "output": ...}` (Alpaca) |
| `chat` | `{"messages": [{"role": ...}, ...]}` (mlx-lm / HF chat) |
| `completion` | `{"prompt": ..., "completion": ...}` |

### Backends (Phase 3)

`student.backend` selects `mlx` (Apple-Silicon-native default) or `peft`
(PyTorch/MPS). Both implement `StudentBackend`; the trainer is identical.

> Memory note: `distill evaluate --before` loads two student instances (untrained
> + trained). On 24 GB, pass `--no-before` if you hit pressure.

## Configuration (Phase 9)

Everything is a single validated `ExperimentConfig` (`configs/experiment_001.yaml`).
Unknown keys are rejected at load time; enums reject invalid choices; a config
snapshot is written into every experiment dir for provenance.

## Extending the framework

Each extension is additive — no core changes:

| Add a… | Do this |
|---|---|
| Distillation strategy (logit/KL, hidden, attention) | Subclass `DistillationStrategy`, `@register_strategy(kind)` in `losses/` |
| Student backend | Implement `StudentBackend`, wire into `student/factory.py` |
| Dataset format | Subclass `FormatAdapter`, add to the registry in `datasets/formats.py` |
| Metric | Subclass `TextMetric`, add to the registry in `evaluation/metrics.py` |

The `LossInputs` container already carries `teacher_logits`, hidden states, and
attentions, so logit/hidden/attention distillation need no new plumbing.

## Development (Phase 10)

```bash
pytest                    # 59 tests; none require Ollama/MLX/torch
ruff check . && ruff format --check .
mypy distillation         # strict mode, clean
pre-commit install        # enable hooks (ruff, ruff-format, mypy)
```

## Roadmap

Response distillation ships today. The interfaces are designed for: logit /
multi-teacher / progressive distillation, speculative decoding, DPO/RLHF/GRPO,
synthetic & RAG dataset generation, and an automated benchmark suite.
# Qwen-model-distillation
