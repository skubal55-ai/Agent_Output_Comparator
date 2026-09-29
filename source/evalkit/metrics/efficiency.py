"""Latency-based efficiency metric.

Unlike the original single-peer comparison, this normalizes latency against
the full set of latencies observed for the same prompt in context
(``peer_latencies_ms``), which generalizes beyond exactly two agents. Falls
back to a neutral score when no peer data is available.
"""
from __future__ import annotations

from typing import Any

from .base import Metric, MetricResult


class EfficiencyMetric(Metric):
    """Min-max normalized latency score (0-100); lower latency scores higher.

    Given ``context["peer_latencies_ms"]`` (all latencies for the same
    prompt across agents/trials, including this one), the fastest gets 100,
    the slowest gets 20, linearly interpolated. With <2 distinct values,
    returns a neutral 70.
    """

    name = "efficiency"

    def compute(self, *, output: str, prompt: str, system_prompt: str = "", **context: Any) -> MetricResult:
        latency_ms = context.get("latency_ms")
        peers = context.get("peer_latencies_ms") or []
        values = [v for v in peers if isinstance(v, (int, float)) and v > 0]
        if latency_ms:
            values = list(values) + [latency_ms]

        if latency_ms is None or len(set(values)) < 2:
            return MetricResult(score=70.0, details={"note": "insufficient peer data; neutral score"})

        lo, hi = min(values), max(values)
        # Faster (lower latency) => higher score.
        fraction_slow = (latency_ms - lo) / (hi - lo)
        score = 100.0 - fraction_slow * 80.0  # 100 fastest .. 20 slowest
        return MetricResult(
            score=max(20.0, min(score, 100.0)),
            details={"latency_ms": latency_ms, "min_ms": lo, "max_ms": hi},
        )
