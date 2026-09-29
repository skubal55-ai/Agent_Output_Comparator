"""Task-success metrics: the strongest correctness signal, when available.

Two pluggable backends are provided:

- TestExecutionMetric: runs a test command against files the agent produced
  and scores by pass rate. This is the recommended backend for coding tasks
  where a test suite can be defined per-prompt.
- LLMJudgeMetric: delegates scoring to an externally-supplied judge callable
  (e.g. a wrapped LLM API call with a fixed rubric prompt). No judge backend
  is wired up by default in this environment (no LLM API key configured
  here) — supply ``judge_fn`` to enable it. Documented in
  docs/methodology.md as requiring a deterministic rubric prompt and
  reporting the judge model/version for reproducibility.
"""
from __future__ import annotations

import subprocess
from typing import Any, Callable

from .base import Metric, MetricResult


class TestExecutionMetric(Metric):
    """Runs a test command and scores by pass rate (0-100).

    ``context["test_command"]`` — list[str], e.g. ["pytest", "-q"].
    ``context["cwd"]`` — working directory to run the command in.
    Requires the test runner to report a parseable pass/fail summary via
    return code as a minimal signal (returncode == 0 -> 100, else 0), which
    callers can refine by supplying ``context["pass_rate"]`` directly
    (0.0-1.0) when a more granular result is already known (e.g. parsed
    JUnit XML) — that takes precedence over running a subprocess.
    """

    name = "task_success_tests"

    def compute(self, *, output: str, prompt: str, system_prompt: str = "", **context: Any) -> MetricResult:
        pass_rate = context.get("pass_rate")
        if pass_rate is not None:
            return MetricResult(score=max(0.0, min(float(pass_rate) * 100.0, 100.0)), details={"pass_rate": pass_rate})

        test_command = context.get("test_command")
        cwd = context.get("cwd")
        if not test_command:
            return MetricResult(
                score=50.0,
                details={"note": "no test_command/pass_rate provided; task success not evaluated (neutral score)"},
            )

        try:
            proc = subprocess.run(
                test_command, cwd=cwd, capture_output=True, text=True, timeout=300
            )
            passed = proc.returncode == 0
            return MetricResult(
                score=100.0 if passed else 0.0,
                details={"returncode": proc.returncode, "stdout_tail": (proc.stdout or "")[-500:]},
            )
        except Exception as exc:
            return MetricResult(score=0.0, details={"error": str(exc)})


class LLMJudgeMetric(Metric):
    """Rubric-based LLM-judge backend (pluggable, disabled unless configured).

    ``judge_fn(output, prompt, system_prompt) -> float`` must return a score
    in [0, 100]. When no ``judge_fn`` is supplied at construction, this
    metric returns a neutral score and flags itself as disabled rather than
    silently fabricating a judgment.
    """

    name = "task_success_llm_judge"

    def __init__(self, judge_fn: Callable[[str, str, str], float] | None = None) -> None:
        self.judge_fn = judge_fn

    def compute(self, *, output: str, prompt: str, system_prompt: str = "", **context: Any) -> MetricResult:
        if self.judge_fn is None:
            return MetricResult(
                score=50.0,
                details={"enabled": False, "note": "no judge_fn configured; neutral score"},
            )
        score = float(self.judge_fn(output, prompt, system_prompt))
        return MetricResult(score=max(0.0, min(score, 100.0)), details={"enabled": True})
