"""Deterministic, label-based retrieval evaluation; no model judging."""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass

from standard_retrieval import StandardRetrieval


@dataclass(frozen=True)
class StandardRetrievalEvaluationItem:
    query: str
    standard_id: str
    revision: str
    category: str
    expected_source_ids: tuple[str, ...] = ()
    expected_section: str | None = None


@dataclass(frozen=True)
class StandardRetrievalMetrics:
    query_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    exact_section_hit_rate: float
    median_latency_ms: float
    p95_latency_ms: float


@dataclass(frozen=True)
class StandardRetrievalEvaluationReport:
    modes: dict[str, StandardRetrievalMetrics]
    by_category: dict[str, dict[str, StandardRetrievalMetrics]]
    embedding_model: str | None
    retrieval_policy: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "modes": {key: asdict(value) for key, value in self.modes.items()},
            "by_category": {category: {mode: asdict(metrics)
                            for mode, metrics in modes.items()}
                            for category, modes in self.by_category.items()},
            "embedding_model": self.embedding_model,
            "retrieval_policy": self.retrieval_policy,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def to_text(self) -> str:
        lines = []

        for mode, metrics in self.modes.items():
            lines.append(
                f"{mode}: R@1={metrics.recall_at_1:.3f} "
                f"R@3={metrics.recall_at_3:.3f} R@5={metrics.recall_at_5:.3f} "
                f"MRR={metrics.mrr:.3f} exact-section="
                f"{metrics.exact_section_hit_rate:.3f}")

        return "\n".join(lines)


def _evaluate_mode(retrieval, items, mode):
    observations = []

    for item in items:
        started = time.perf_counter()
        results = retrieval.search(item.standard_id, item.revision, item.query,
                                   mode=mode, limit=5,
                                   expand_cross_references=True)
        elapsed = (time.perf_counter() - started) * 1000
        relevant_ranks = [result.rank for result in results
                          if (result.source_id in item.expected_source_ids
                              or (item.expected_section is not None
                                  and result.section == item.expected_section))]

        if not item.expected_source_ids and item.expected_section is None:
            rank = 1 if not results else None
        else:
            rank = min(relevant_ranks) if relevant_ranks else None

        observations.append((item, rank, elapsed))

    count = len(observations)
    structural = [(item, rank) for item, rank, _ in observations
                  if item.category == "structural"]
    latencies = sorted(elapsed for _, _, elapsed in observations)
    p95_index = min(len(latencies) - 1, max(0, int(.95 * len(latencies))))

    return StandardRetrievalMetrics(
        count,
        sum(rank is not None and rank <= 1 for _, rank, _ in observations) / count,
        sum(rank is not None and rank <= 3 for _, rank, _ in observations) / count,
        sum(rank is not None and rank <= 5 for _, rank, _ in observations) / count,
        sum(1 / rank if rank else 0 for _, rank, _ in observations) / count,
        (sum(rank == 1 for _, rank in structural) / len(structural)
         if structural else 0.0),
        round(statistics.median(latencies), 6), round(latencies[p95_index], 6),
    )


def evaluate_retrieval(retrieval: StandardRetrieval,
                       items: tuple[StandardRetrievalEvaluationItem, ...], *,
                       modes=("lexical", "vector", "hybrid")):
    if not items:
        raise ValueError("retrieval evaluation requires labeled items")

    overall = {mode: _evaluate_mode(retrieval, items, mode) for mode in modes}
    categories = {}

    for category in sorted({item.category for item in items}):
        selected = tuple(item for item in items if item.category == category)
        categories[category] = {
            mode: _evaluate_mode(retrieval, selected, mode) for mode in modes}

    return StandardRetrievalEvaluationReport(
        overall, categories,
        retrieval.embedder.model_id if retrieval.embedder else None,
        dict(retrieval.policy.__dict__),
    )
