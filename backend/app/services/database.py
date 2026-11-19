"""SQLite Database service for GraphGuard inference logging."""
from __future__ import annotations

import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger("backend.database")


class Database:
    def __init__(self, db_path: str | Path = "data/inference.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    tx_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    latency_ms REAL NOT NULL
                )
                """
            )
            conn.commit()

    def log_inference(
        self,
        tx_id: str,
        model_id: str,
        score: float,
        latency_ms: float,
    ) -> None:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO inference_log (ts, tx_id, model_id, score, latency_ms)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, tx_id, model_id, score, latency_ms),
            )
            conn.commit()

    def get_recent_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT id, ts, tx_id, model_id, score, latency_ms
                FROM inference_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]

