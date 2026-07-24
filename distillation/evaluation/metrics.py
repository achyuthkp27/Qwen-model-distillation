"""Text-similarity metrics (Phase 5).

Each metric implements :class:`TextMetric` and is discoverable via a registry, so
the evaluator asks for metrics by name (from config) and never imports a metric
library directly. Heavy dependencies (sacrebleu, rouge-score, bert-score) are
imported lazily inside ``compute`` so the framework loads without the ``eval``
extra installed; a clear error names the missing extra if a metric is requested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from distillation.utils.logging import get_logger

_logger = get_logger(__name__)


class TextMetric(ABC):
    """A metric comparing generated predictions against references."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Registry name of this metric."""

    @abstractmethod
    def compute(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Return one or more scalar scores for the prediction/reference pairs."""


class BleuMetric(TextMetric):
    """Corpus BLEU via ``sacrebleu`` (0-100)."""

    @property
    def name(self) -> str:
        """The metric name."""
        return "bleu"

    def compute(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Compute corpus BLEU."""
        try:
            import sacrebleu
        except ImportError as exc:  # pragma: no cover - env-dependent.
            raise ImportError("bleu requires: pip install -e '.[eval]'") from exc
        score = sacrebleu.corpus_bleu(predictions, [references]).score
        return {"bleu": float(score)}


class RougeMetric(TextMetric):
    """ROUGE-1/2/L F-measures via ``rouge_score`` (averaged over samples)."""

    @property
    def name(self) -> str:
        """The metric name."""
        return "rouge"

    def compute(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Compute mean ROUGE-1/2/L F1."""
        try:
            from rouge_score import rouge_scorer
        except ImportError as exc:  # pragma: no cover - env-dependent.
            raise ImportError("rouge requires: pip install -e '.[eval]'") from exc
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for pred, ref in zip(predictions, references, strict=True):
            scores = scorer.score(ref, pred)
            for key in totals:
                totals[key] += scores[key].fmeasure
        n = max(len(predictions), 1)
        return {k: v / n for k, v in totals.items()}


class BertScoreMetric(TextMetric):
    """BERTScore F1 via ``bert_score``.

    Args:
        model_type: The HF model backing BERTScore embeddings.
    """

    def __init__(self, model_type: str = "microsoft/deberta-xlarge-mnli") -> None:
        self._model_type = model_type

    @property
    def name(self) -> str:
        """The metric name."""
        return "bertscore"

    def compute(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Compute mean BERTScore F1."""
        try:
            from bert_score import score as bert_score
        except ImportError as exc:  # pragma: no cover - env-dependent.
            raise ImportError("bertscore requires: pip install -e '.[eval]'") from exc
        _, _, f1 = bert_score(predictions, references, model_type=self._model_type, verbose=False)
        return {"bertscore_f1": float(f1.mean().item())}


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------
#: Names handled by the evaluator itself (measured during generation), not here.
PERFORMANCE_METRICS = frozenset({"latency", "tokens_per_sec", "memory", "memory_usage"})

_TEXT_METRICS: dict[str, type[TextMetric]] = {
    "bleu": BleuMetric,
    "rouge": RougeMetric,
    "bertscore": BertScoreMetric,
}


def get_text_metrics(
    names: list[str], *, bertscore_model: str = "microsoft/deberta-xlarge-mnli"
) -> list[TextMetric]:
    """Instantiate the requested text metrics, ignoring performance-only names.

    Args:
        names: Metric names from the evaluation config.
        bertscore_model: Model id passed to BERTScore if requested.

    Returns:
        Instantiated :class:`TextMetric` objects (performance names filtered out).

    Raises:
        KeyError: If a name is neither a known text metric nor a performance one.
    """
    metrics: list[TextMetric] = []
    for name in names:
        if name in PERFORMANCE_METRICS:
            continue
        if name not in _TEXT_METRICS:
            raise KeyError(
                f"Unknown metric '{name}'. Known: "
                f"{sorted(_TEXT_METRICS) + sorted(PERFORMANCE_METRICS)}."
            )
        if name == "bertscore":
            metrics.append(BertScoreMetric(bertscore_model))
        else:
            metrics.append(_TEXT_METRICS[name]())
    return metrics
