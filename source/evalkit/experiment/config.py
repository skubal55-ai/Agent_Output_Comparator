"""Config-driven experiment definition: agents x prompts x n_trials.

Loadable from YAML (if pyyaml is installed) or a plain dict/JSON, so
experiments are reproducible and diffable rather than typed into a form.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


@dataclass
class PromptSpec:
    id: str
    system_prompt: str = ""
    user_prompt: str = ""
    # Optional task-success grounding for this prompt (see metrics/task_success.py).
    # test_command runs after the agent responds, in test_cwd, and scores by pass/fail;
    # pass_rate — a pre-computed 0.0-1.0 pass rate — takes precedence if both are set.
    test_command: list[str] | None = None
    test_cwd: str | None = None
    pass_rate: float | None = None


@dataclass
class ExperimentConfig:
    name: str
    agents: list[str]
    prompts: list[PromptSpec]
    n_trials: int = 1
    metric_weights: dict[str, float] = field(default_factory=dict)
    agent_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    # >0 enables TestExecutionMetric in the scorer with this weight (normalized
    # alongside the other metric weights). 0 (default) keeps task success out of
    # the aggregate — see docs/methodology.md Section 3/8 on why this must be
    # explicit rather than silently assumed.
    task_success_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        prompts: list[PromptSpec] = []
        for i, p in enumerate(data.get("prompts", [])):
            if isinstance(p, str):
                prompts.append(PromptSpec(id=f"p{i}", user_prompt=p))
            else:
                prompts.append(
                    PromptSpec(
                        id=p.get("id", f"p{i}"),
                        system_prompt=p.get("system_prompt", ""),
                        user_prompt=p.get("user_prompt", p.get("prompt", "")),
                        test_command=p.get("test_command"),
                        test_cwd=p.get("test_cwd") or p.get("cwd"),
                        pass_rate=p.get("pass_rate"),
                    )
                )
        agents = data.get("agents", [])
        if not agents:
            raise ValueError("ExperimentConfig requires at least one agent")
        if not prompts:
            raise ValueError("ExperimentConfig requires at least one prompt")
        n_trials = int(data.get("n_trials", 1))
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        return cls(
            name=data.get("name", "experiment"),
            agents=agents,
            prompts=prompts,
            n_trials=n_trials,
            metric_weights=data.get("metric_weights", {}),
            agent_kwargs=data.get("agent_kwargs", {}),
            task_success_weight=float(data.get("task_success_weight", 0.0)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        text = Path(path).read_text(encoding="utf-8")
        if yaml is not None:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)  # fallback: accept JSON if pyyaml isn't installed
        return cls.from_dict(data)
