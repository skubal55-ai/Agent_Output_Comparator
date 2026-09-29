"""Structural signal metrics, migrated from the original single-function
`score_output()` heuristic in server.py. Kept as minor, cheap signals in the
aggregate score (not the primary correctness/accuracy signal) — see
docs/methodology.md for the rationale and known limitations of proxying
"quality" with surface structure.
"""
from __future__ import annotations

import re
from typing import Any

from .base import Metric, MetricResult

_ERROR_PHRASES = ("error:", "not found", "command not found", "traceback", "exception")
_STRUCTURE_RE = re.compile(r"(^[-*•]\s|\d+\.\s|```|#{1,3}\s)", re.MULTILINE)


class QualityHeuristicMetric(Metric):
    """Surface-level structure/completeness signal (0-100).

    Formula (additive, capped at 100):
      +20  word_count >= 10
      +15  word_count >= 30
      +20  has list/code-block/heading markup
      +15  >= 3 sentences
      +15  no error-like phrases present
      +15  output not truncated (doesn't end with "..."/"...")
    """

    name = "quality_heuristic"

    def compute(self, *, output: str, prompt: str, system_prompt: str = "", **context: Any) -> MetricResult:
        if not output:
            return MetricResult(score=0.0, details={"word_count": 0})

        words = output.split()
        word_count = len(words)
        quality = 0
        if word_count >= 10:
            quality += 20
        if word_count >= 30:
            quality += 15
        has_structure = bool(_STRUCTURE_RE.search(output))
        if has_structure:
            quality += 20
        sentences = re.split(r"[.!?]\s+", output)
        if len(sentences) >= 3:
            quality += 15
        has_error_phrase = any(p in output.lower() for p in _ERROR_PHRASES)
        if not has_error_phrase:
            quality += 15
        truncated = output.rstrip().endswith(("...", "…"))
        if not truncated:
            quality += 15
        quality = min(quality, 100)

        return MetricResult(
            score=float(quality),
            details={
                "word_count": word_count,
                "has_structure": has_structure,
                "sentence_count": len(sentences),
                "has_error_phrase": has_error_phrase,
                "truncated": truncated,
            },
        )


class LengthFitMetric(Metric):
    """Penalizes very short or excessively long responses (0-100).

    Bucketed by word count; ideal range is 30-400 words. This is a coarse
    proxy for verbosity fit, not a substitute for task-specific length
    expectations.
    """

    name = "length_fit"

    def compute(self, *, output: str, prompt: str, system_prompt: str = "", **context: Any) -> MetricResult:
        word_count = len(output.split()) if output else 0
        if word_count < 5:
            score = 10
        elif word_count < 15:
            score = 40
        elif word_count < 30:
            score = 65
        elif word_count <= 400:
            score = 100
        elif word_count <= 700:
            score = 80
        else:
            score = 60
        return MetricResult(score=float(score), details={"word_count": word_count})
