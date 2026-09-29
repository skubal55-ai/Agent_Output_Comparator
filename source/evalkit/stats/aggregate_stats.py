"""Aggregate statistics (mean, stdev, 95% CI) across trials for one
(agent, metric) pair. Uses the t-distribution for the CI since experiment
trial counts are typically small (n < 30).
"""
from __future__ import annotations

import math
from typing import Sequence

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None


def mean_stdev(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def confidence_interval_95(
    values: Sequence[float], bounds: tuple[float, float] | None = None
) -> tuple[float, float]:
    """95% CI for the mean using the t-distribution (falls back to a normal
    approximation, z=1.96, if scipy isn't installed).

    ``bounds`` — the metric's valid range, e.g. (0, 100). With few trials the
    t-interval of a bounded score can extend past the range (e.g. 120.9 for a
    0-100 metric); when given, the interval is truncated to it.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n < 2:
        m = values[0]
        return (m, m)
    m, sd = mean_stdev(values)
    se = sd / math.sqrt(n)
    if _scipy_stats is not None:
        t_crit = float(_scipy_stats.t.ppf(0.975, df=n - 1))
    else:
        t_crit = 1.96
    margin = t_crit * se
    lo, hi = m - margin, m + margin
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return (lo, hi)


def summarize(values: Sequence[float], bounds: tuple[float, float] | None = None) -> dict:
    m, sd = mean_stdev(values)
    lo, hi = confidence_interval_95(values, bounds)
    return {"n": len(values), "mean": m, "stdev": sd, "ci95_low": lo, "ci95_high": hi}
