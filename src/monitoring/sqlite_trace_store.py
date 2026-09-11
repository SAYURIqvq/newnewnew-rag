"""Small local trace store for single-machine RAG observability."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteTraceStore:
    """Persist local request traces without requiring a telemetry service."""

    def __init__(self, path: str = "data/observability/rag_traces.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS request_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    query_preview TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    role TEXT NOT NULL,
                    strategy TEXT,
                    latency_seconds REAL NOT NULL,
                    chunks_used INTEGER NOT NULL,
                    citations INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def record(
        self,
        *,
        query: str,
        mode: str,
        role: str,
        strategy: str,
        latency_seconds: float,
        chunks_used: int,
        citations: int,
        metadata: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO request_traces
                    (query_preview, mode, role, strategy, latency_seconds,
                     chunks_used, citations, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query[:240],
                    mode,
                    role,
                    strategy,
                    latency_seconds,
                    chunks_used,
                    citations,
                    json.dumps(metadata, ensure_ascii=False, default=str),
                ),
            )

    def summary(self) -> dict[str, float | int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT latency_seconds FROM request_traces ORDER BY latency_seconds"
            ).fetchall()
        latencies = [row[0] for row in rows]
        if not latencies:
            return {"count": 0, "average": 0.0, "p50": 0.0, "p95": 0.0}

        def percentile(value: float) -> float:
            index = max(0, min(len(latencies) - 1, round((len(latencies) - 1) * value)))
            return float(latencies[index])

        return {
            "count": len(latencies),
            "average": round(sum(latencies) / len(latencies), 3),
            "p50": round(percentile(0.50), 3),
            "p95": round(percentile(0.95), 3),
        }

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, query_preview, mode, role, strategy,
                       latency_seconds, chunks_used, citations, metadata_json
                FROM request_traces
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "time": row[0], "query": row[1], "mode": row[2], "role": row[3],
                "strategy": row[4], "latency_s": row[5], "chunks": row[6],
                "citations": row[7], "details": json.loads(row[8]),
            }
            for row in rows
        ]

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM request_traces")
