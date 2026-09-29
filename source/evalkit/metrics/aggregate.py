"""AggregateScorer: combines an arbitrary set of Metric instances into a
single weighted overall score, with configurable weights (not hardcoded),
so a paper can report a weight-sensitivity analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Metric, MetricResult
from .structure import QualityHeuristicMetric, LengthFitMetric
from .semantic_accuracy import SemanticAccuracyMetric
from .efficiency import EfficiencyMetric
from .task_success import TestExecutionMetric


@dataclass
class WeightedMetric:
    key: str
    metric: Metric
    weight: float


class AggregateScorer:
    """Weighted combination of independent metrics.

    Weights need not sum to 1; they are normalized at combine time so
    metrics can be added/removed without re-deriving every other weight.
    """

    def __init__(self, weighted_metrics: list[WeightedMetric]) -> None:
        if not weighted_metrics:
            raise ValueError("AggregateScorer requires at least one metric")
        self.weighted_metrics = weighted_metrics

    def score(
        self,
        *,
        output: str,
        prompt: str,
        system_prompt: str = "",
        **context: Any,
    ) -> dict[str, Any]:
        results: dict[str, MetricResult] = {}
        for wm in self.weighted_metrics:
            results[wm.key] = wm.metric.compute(
                output=output, prompt=prompt, system_prompt=system_prompt, **context
            )

        total_weight = sum(wm.weight for wm in self.weighted_metrics) or 1.0
        overall = sum(
            results[wm.key].score * (wm.weight / total_weight) for wm in self.weighted_metrics
        )

        out: dict[str, Any] = {
            wm.key: {"score": results[wm.key].score, "details": results[wm.key].details}
            for wm in self.weighted_metrics
        }
        out["overall"] = round(overall, 2)
        out["word_count"] = len(output.split()) if output else 0
        out["weights"] = {wm.key: wm.weight / total_weight for wm in self.weighted_metrics}
        out["failed"] = False
        return out

    def failed_result(self, error: str) -> dict[str, Any]:
        """Scores for an agent run that errored (auth failure, timeout, crash).

        Same shape as ``score()`` but every metric is 0 and ``failed`` is set.
        Scoring the error text instead would reward failures — e.g. a fast
        auth error earns full efficiency credit.
        """
        total_weight = sum(wm.weight for wm in self.weighted_metrics) or 1.0
        out: dict[str, Any] = {
            wm.key: {"score": 0.0, "details": {"error": error}} for wm in self.weighted_metrics
        }
        out["overall"] = 0.0
        out["word_count"] = 0
        out["weights"] = {wm.key: wm.weight / total_weight for wm in self.weighted_metrics}
        out["failed"] = True
        return out


def default_scorer(task_success_weight: float = 0.0) -> AggregateScorer:
    """The framework's default metric set and weights.

    Weights mirror the original heuristic's emphasis (quality + accuracy as
    primary signals, speed/length as secondary) but are now named,
    independently swappable, and passed explicitly rather than hardcoded
    inside a single scoring function. See docs/methodology.md for the
    rationale and the validation results these weights should be checked
    against.

    ``task_success_weight`` — when > 0, adds TestExecutionMetric (the
    strongest correctness signal, see metrics/task_success.py) to the
    weighted set with this weight; all weights are renormalized together.
    Left at 0 by default because task success requires per-prompt test
    definitions the framework cannot assume are available (see
    docs/methodology.md Section 8).
    """
    weighted = [
        WeightedMetric("quality", QualityHeuristicMetric(), 0.30),
        WeightedMetric("semantic_accuracy", SemanticAccuracyMetric(), 0.35),
        WeightedMetric("efficiency", EfficiencyMetric(), 0.15),
        WeightedMetric("length_fit", LengthFitMetric(), 0.20),
    ]
    if task_success_weight > 0:
        weighted.append(WeightedMetric("task_success", TestExecutionMetric(), task_success_weight))
    return AggregateScorer(weighted)
