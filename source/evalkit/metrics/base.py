"""Metric interface. Every metric is independent, documents its own formula,
and returns a MetricResult so AggregateScorer can combine an arbitrary set
of them without knowing their internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """A single metric's output. score is always normalized to 0-100."""

    score: float
    details: dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    """A single, independently-computable evaluation dimension."""

    name: str = "metric"

    @abstractmethod
    def compute(
        self,
        *,
        output: str,
        prompt: str,
        system_prompt: str = "",
        **context: Any,
    ) -> MetricResult:
        """Score ``output`` for the given prompt. ``context`` carries optional
        cross-cutting info a metric may use (e.g. ``peer_latencies_ms`` for
        EfficiencyMetric, ``cwd``/``test_command`` for TestExecutionMetric).
        Unused context keys must be ignored, not raise.
        """
        raise NotImplementedError
