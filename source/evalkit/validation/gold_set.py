"""Schema and CSV I/O for a human-labeled gold set used to validate
automated metrics (see correlate.py). This module does not fabricate
human judgments — it only loads/saves what a human rater supplies.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class GoldExample:
    id: str
    prompt: str
    output: str
    human_score: float  # 0-100, this rater's judgment of `output` for `prompt`
    rater_id: str
    system_prompt: str = ""


_FIELDNAMES = ["id", "prompt", "system_prompt", "output", "human_score", "rater_id"]


def load_gold_set_csv(path: str | Path) -> list[GoldExample]:
    examples: list[GoldExample] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            examples.append(
                GoldExample(
                    id=row["id"],
                    prompt=row["prompt"],
                    output=row["output"],
                    human_score=float(row["human_score"]),
                    rater_id=row["rater_id"],
                    system_prompt=row.get("system_prompt", "") or "",
                )
            )
    return examples


def save_gold_set_csv(examples: list[GoldExample], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for ex in examples:
            writer.writerow(asdict(ex))
