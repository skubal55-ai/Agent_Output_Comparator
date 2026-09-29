"""SQLite persistence for experiments and trials.

Kept deliberately simple (two tables, JSON columns for nested data) — this
is a local research tool, not a service with concurrent writers.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# source/evalkit/experiment/storage.py -> repo root
DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "evalkit_results.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    trial_n INTEGER NOT NULL,
    output TEXT,
    latency_ms INTEGER,
    error TEXT,
    scores_json TEXT NOT NULL,
    reproducibility_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX IF NOT EXISTS idx_trials_experiment ON trials(experiment_id);
"""


class ResultStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create_experiment(self, name: str, config: dict[str, Any]) -> str:
        experiment_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO experiments (id, name, config_json, created_at) VALUES (?, ?, ?, ?)",
            (experiment_id, name, json.dumps(config), datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        return experiment_id

    def add_trial(
        self,
        *,
        experiment_id: str,
        agent: str,
        prompt_id: str,
        trial_n: int,
        output: str,
        latency_ms: int,
        error: str | None,
        scores: dict[str, Any],
        reproducibility: dict[str, Any],
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO trials
              (experiment_id, agent, prompt_id, trial_n, output, latency_ms, error,
               scores_json, reproducibility_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                agent,
                prompt_id,
                trial_n,
                output,
                latency_ms,
                error,
                json.dumps(scores),
                json.dumps(reproducibility),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, name, config_json, created_at FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        exp = {"id": row[0], "name": row[1], "config": json.loads(row[2]), "created_at": row[3]}
        exp["trials"] = self.get_trials(experiment_id)
        return exp

    def get_trials(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT agent, prompt_id, trial_n, output, latency_ms, error,
                   scores_json, reproducibility_json, created_at
            FROM trials WHERE experiment_id = ? ORDER BY prompt_id, trial_n, agent
            """,
            (experiment_id,),
        ).fetchall()
        return [
            {
                "agent": r[0],
                "prompt_id": r[1],
                "trial_n": r[2],
                "output": r[3],
                "latency_ms": r[4],
                "error": r[5],
                "scores": json.loads(r[6]),
                "reproducibility": json.loads(r[7]),
                "created_at": r[8],
            }
            for r in rows
        ]

    def list_experiments(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, name, created_at FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]

    def close(self) -> None:
        self._conn.close()
