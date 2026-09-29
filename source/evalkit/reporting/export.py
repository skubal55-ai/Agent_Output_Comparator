"""Flatten a stored experiment (agent x prompt x trial rows, each with
nested metric scores) into CSV/JSON suitable for paper tables and external
statistical analysis (R, pandas, spreadsheets).
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any


def _flatten_trial(trial: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "agent": trial["agent"],
        "prompt_id": trial["prompt_id"],
        "trial_n": trial["trial_n"],
        "latency_ms": trial["latency_ms"],
        "error": trial["error"] or "",
        "overall_score": trial["scores"].get("overall"),
        "agent_version": trial["reproducibility"].get("agent_version"),
        "os_platform": trial["reproducibility"].get("os_platform"),
        "timestamp_utc": trial["reproducibility"].get("timestamp_utc"),
    }
    for key, val in trial["scores"].items():
        if isinstance(val, dict) and "score" in val:
            flat[f"metric_{key}"] = val["score"]
    return flat


def experiment_to_json(experiment: dict[str, Any]) -> str:
    return json.dumps(experiment, indent=2)


def experiment_to_csv(experiment: dict[str, Any]) -> str:
    rows = [_flatten_trial(t) for t in experiment.get("trials", [])]
    if not rows:
        return ""
    fieldnames = sorted({k for row in rows for k in row})
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()
