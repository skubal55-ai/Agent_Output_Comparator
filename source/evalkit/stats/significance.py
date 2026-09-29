"""Paired significance testing between two agents on the same prompts, and
Holm-Bonferroni correction for comparing more than two agents pairwise.

Default is the Wilcoxon signed-rank test: non-parametric, appropriate for
bounded 0-100 scores that aren't guaranteed to be normally distributed.
A paired t-test is offered as an alternative for larger, more normal
samples.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None


def rank_biserial(diffs: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation (Kerby, 2014): (R+ - R-) / (R+ + R-),
    where R+ / R- are the rank sums of |d| over the positive / negative non-zero
    paired differences. Ranges from -1 to 1; zero differences are dropped, as in
    the Wilcoxon signed-rank test itself."""
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return 0.0
    ranks = _scipy_stats.rankdata([abs(d) for d in nonzero])
    r_pos = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    r_neg = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    return float((r_pos - r_neg) / (r_pos + r_neg))


def paired_significance(a: Sequence[float], b: Sequence[float], method: str = "wilcoxon") -> dict:
    """Compare paired score samples ``a`` vs ``b`` (same prompts, two agents).

    Returns statistic, p_value, effect_size (matched-pairs rank-biserial
    correlation for Wilcoxon, see rank_biserial), and n. Raises if scipy isn't
    installed or inputs are invalid.
    """
    if _scipy_stats is None:
        raise ImportError("scipy is required for significance testing; pip install scipy")
    if len(a) != len(b):
        raise ValueError("paired_significance requires equal-length paired samples")
    if len(a) < 2:
        raise ValueError("paired_significance requires at least 2 paired observations")

    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        return {"method": method, "statistic": 0.0, "p_value": 1.0, "effect_size": 0.0, "n": len(a)}

    if method == "ttest":
        statistic, p_value = _scipy_stats.ttest_rel(a, b)
        effect_size = float(statistic) / (len(a) ** 0.5)
    else:
        statistic, p_value = _scipy_stats.wilcoxon(a, b)
        effect_size = rank_biserial(diffs)

    return {
        "method": method,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "n": len(a),
    }


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Given m p-values from m separate tests, controls the family-wise error
    rate at ``alpha`` (stronger than uncorrected per-test alpha, less
    conservative than plain Bonferroni). Returns one dict per input p-value,
    in input order: {p_value, rank, threshold, significant}.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    results: list[dict | None] = [None] * m
    still_rejecting = True
    for rank, idx in enumerate(order, start=1):
        threshold = alpha / (m - rank + 1)
        significant = still_rejecting and p_values[idx] <= threshold
        if not significant:
            still_rejecting = False
        results[idx] = {
            "p_value": p_values[idx],
            "rank": rank,
            "threshold": threshold,
            "significant": significant,
        }
    return results  # type: ignore[return-value]


def all_pairs_significance(
    scores_by_agent: dict[str, Sequence[float]], method: str = "wilcoxon", alpha: float = 0.05
) -> dict:
    """Pairwise significance across two or more agents, with Holm-Bonferroni
    correction applied across all pairs once there are more than two.

    Every agent's sequence must have the same length and be paired by
    position (e.g. same prompt/trial order) — this is the caller's
    responsibility, same as ``paired_significance``. Comparing more than
    two agents pairwise without this correction inflates the family-wise
    false-positive rate (see docs/methodology.md, Threats to Validity).
    """
    agents = sorted(scores_by_agent)
    pairs = list(combinations(agents, 2))
    if not pairs:
        return {"alpha": alpha, "pairs": []}

    raw = [(a, b, paired_significance(scores_by_agent[a], scores_by_agent[b], method=method)) for a, b in pairs]
    corrected = holm_bonferroni([r[2]["p_value"] for r in raw], alpha=alpha) if len(raw) > 1 else None

    out = []
    for i, (a, b, result) in enumerate(raw):
        entry = {"agent_a": a, "agent_b": b, **result}
        if corrected is not None:
            entry["holm_bonferroni"] = corrected[i]
        out.append(entry)
    return {"alpha": alpha, "pairs": out}
