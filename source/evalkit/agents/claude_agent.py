"""Agent wrapper around the Claude Code CLI runner in server.py.

The subprocess/JSON-parsing logic lives in server.py alongside the Copilot
runner; this module only adapts its dict return value into the framework's
AgentResult shape.
"""
from __future__ import annotations

import subprocess
from typing import Any

from .base import Agent, AgentResult


class ClaudeCodeAgent(Agent):
    name = "claude"

    def __init__(self) -> None:
        self._version_cache: str | None = None

    def run(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> AgentResult:
        # Imported lazily to avoid a circular import (server.py also imports evalkit).
        from server import run_claude_cli

        result = run_claude_cli(system_prompt, user_prompt, cwd=kwargs.get("cwd"))
        meta = {k: v for k, v in result.items() if k not in ("output", "latency_ms", "error")}
        meta["agent_version"] = self.version()
        return AgentResult(
            output=result.get("output", "") or "",
            latency_ms=int(result.get("latency_ms", 0) or 0),
            error=result.get("error"),
            meta=meta,
        )

    def version(self) -> str:
        if self._version_cache is not None:
            return self._version_cache
        from server import _find_claude_binary

        exe = _find_claude_binary() or "claude"
        try:
            proc = subprocess.run(
                [exe, "--version"], capture_output=True, text=True, timeout=15, shell=False
            )
            out = (proc.stdout or proc.stderr or "").strip()
            self._version_cache = out.splitlines()[0] if out else "unknown"
        except Exception:
            self._version_cache = "unknown"
        return self._version_cache
