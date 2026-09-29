"""Captures per-trial reproducibility metadata: agent CLI version, OS,
Python version, and timestamp — so results can be tied to the exact
environment they were produced in.
"""
from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class ReproducibilityMetadata:
    agent_name: str
    agent_version: str
    os_platform: str
    python_version: str
    timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture(agent_name: str, agent_version: str) -> ReproducibilityMetadata:
    return ReproducibilityMetadata(
        agent_name=agent_name,
        agent_version=agent_version,
        os_platform=platform.platform(),
        python_version=sys.version.split()[0],
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
