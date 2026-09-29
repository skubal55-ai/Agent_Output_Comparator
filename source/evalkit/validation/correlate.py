"""Validate an automated Metric against human judgment on a gold set:
correlation between metric score and human score, and inter-rater
agreement between two human raters.

This is the tooling for the paper's metric-validation section. It does not
generate or fabricate human judgments — the gold set must be supplied.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None

from ..metrics.base import Metric
from .gold_set import GoldExample


def metric_correlation(examples: Sequence[GoldExample], metric: Metric) -> dict:
    """Correlate ``metric``'s automated score against human_score across the
    gold set. When multiple raters scored the same example id, their scores
    are averaged before correlating.
    """
    if _scipy_stats is None:
        raise ImportError("scipy is required for correlation analysis; pip install scipy")

    by_id: dict[str, list[GoldExample]] = defaultdict(list)
    for ex in examples:
        by_id[ex.id].append(ex)

    automated_scores = []
    human_scores = []
    for group in by_id.values():
        rep = group[0]
        result = metric.compute(output=rep.output, prompt=rep.prompt, system_prompt=rep.system_prompt)
        automated_scores.append(result.score)
        human_scores.append(sum(g.human_score for g in group) / len(group))

    if len(automated_scores) < 3:
        raise ValueError("Need at least 3 gold examples to compute a meaningful correlation")

    pearson_r, pearson_p = _scipy_stats.pearsonr(automated_scores, human_scores)
    spearman_r, spearman_p = _scipy_stats.spearmanr(automated_scores, human_scores)

    return {
        "n": len(automated_scores),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


def inter_rater_agreement(
    examples: Sequence[GoldExample], score_bin_edges: Sequence[float] = (34, 67)
) -> dict:
    """Cohen's kappa between exactly two raters. human_score is continuous
    (0-100); kappa needs categorical labels, so scores are bucketed into
    low/medium/high using ``score_bin_edges`` before comparing.
    """
    by_rater: dict[str, dict[str, float]] = defaultdict(dict)
    for ex in examples:
        by_rater[ex.rater_id][ex.id] = ex.human_score

    rater_ids = sorted(by_rater)
    if len(rater_ids) != 2:
        raise ValueError(f"inter_rater_agreement requires exactly 2 raters, found {len(rater_ids)}")

    r1, r2 = rater_ids
    shared_ids = sorted(set(by_rater[r1]) & set(by_rater[r2]))
    if len(shared_ids) < 3:
        raise ValueError("Need at least 3 examples rated by both raters")

    def bucket(score: float) -> str:
        lo, hi = score_bin_edges
        if score < lo:
            return "low"
        if score < hi:
            return "medium"
        return "high"

    labels1 = [bucket(by_rater[r1][i]) for i in shared_ids]
    labels2 = [bucket(by_rater[r2][i]) for i in shared_ids]

    kappa = _cohens_kappa(labels1, labels2)
    return {"rater_a": r1, "rater_b": r2, "n": len(shared_ids), "cohens_kappa": kappa}


def _cohens_kappa(labels1: Sequence[str], labels2: Sequence[str]) -> float:
    categories = sorted(set(labels1) | set(labels2))
    n = len(labels1)
    if n == 0:
        return 0.0
    observed_agreement = sum(1 for a, b in zip(labels1, labels2) if a == b) / n

    p1 = {c: labels1.count(c) / n for c in categories}
    p2 = {c: labels2.count(c) / n for c in categories}
    expected_agreement = sum(p1[c] * p2[c] for c in categories)

    if expected_agreement == 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)
