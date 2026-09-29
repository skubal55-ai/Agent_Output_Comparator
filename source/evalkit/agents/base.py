"""Agent interface every evaluated coding agent must implement.

Adding a new agent to the framework (e.g. a third CLI) means implementing
this one interface — metrics, the experiment runner, persistence, and
statistics do not need to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    """Normalized result of a single agent invocation."""

    output: str
    latency_ms: int
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Common interface for a coding agent under evaluation."""

    name: str = "agent"

    @abstractmethod
    def run(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> AgentResult:
        """Execute the agent against a prompt and return a normalized result."""
        raise NotImplementedError

    def version(self) -> str:
        """Tool/CLI version string, captured for reproducibility metadata."""
        return "unknown"
