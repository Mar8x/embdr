"""
File-level SQLite metadata database for ingest tracking.

Separate from ChromaDB — stores per-file stats (char_count, chunk_count),
contextual generation progress, and Ollama benchmark data.

Located at ``db_dir / "embd_meta.db"`` alongside the ``chroma_db/`` directory.
"""
from __future__ import annotations

import hashlib
import logging
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _machine_id() -> str:
    """Short hash of hostname + processor for Ollama benchmark keying."""
    raw = f"{platform.node()}{platform.processor()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class MetaDB:
    """File-level metadata stored in a standalone SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        logger.debug("MetaDB ready: %s", db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    _FILES_DDL = """\
CREATE TABLE IF NOT EXISTS files (
    source_key          TEXT PRIMARY KEY,
    file_hash           TEXT,
    mtime               REAL,
    char_count          INTEGER,
    token_count         INTEGER,
    chunk_count         INTEGER,
    contextual_done     INTEGER DEFAULT 0,
    contextual_at       TEXT,
    contextual_backend  TEXT,
    context_tokens_used INTEGER DEFAULT 0,
    context_cost_usd    REAL DEFAULT 0.0,
    used_sliding_window INTEGER DEFAULT 0,
    source_missing      INTEGER DEFAULT 0
)"""

    _BENCHMARKS_DDL = """\
CREATE TABLE IF NOT EXISTS ollama_benchmarks (
    model           TEXT,
    machine_id      TEXT,
    tok_per_sec     REAL,
    sample_chunks   INTEGER,
    measured_at     TEXT,
    PRIMARY KEY (model, machine_id)
)"""

    # Columns that may not exist in older databases — added idempotently.
    _FILES_COLUMNS = [
        ("file_hash", "TEXT"),
        ("mtime", "REAL"),
        ("char_count", "INTEGER"),
        ("token_count", "INTEGER"),
        ("chunk_count", "INTEGER"),
        ("contextual_done", "INTEGER DEFAULT 0"),
        ("contextual_at", "TEXT"),
        ("contextual_backend", "TEXT"),
        ("context_tokens_used", "INTEGER DEFAULT 0"),
        ("context_cost_usd", "REAL DEFAULT 0.0"),
        ("used_sliding_window", "INTEGER DEFAULT 0"),
        ("source_missing", "INTEGER DEFAULT 0"),
    ]

    def _migrate(self) -> None:
        """Idempotent schema creation and migration."""
        cur = self._conn.cursor()
        cur.execute(self._FILES_DDL)
        cur.execute(self._BENCHMARKS_DDL)

        # Add any missing columns to an existing files table.
        for col_name, col_type in self._FILES_COLUMNS:
            try:
                cur.execute(f"ALTER TABLE files ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass  # column already exists

        self._conn.commit()

    # ------------------------------------------------------------------
    # File CRUD
    # ------------------------------------------------------------------

    def upsert_file(
        self,
        source_key: str,
        file_hash: str,
        mtime: float,
        char_count: int,
        chunk_count: int,
    ) -> None:
        """Insert or update a file's stats.  ``token_count`` = ``char_count // 4``."""
        self._conn.execute(
            """\
INSERT INTO files (source_key, file_hash, mtime, char_count, token_count, chunk_count)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(source_key) DO UPDATE SET
    file_hash   = excluded.file_hash,
    mtime       = excluded.mtime,
    char_count  = excluded.char_count,
    token_count = excluded.token_count,
    chunk_count = excluded.chunk_count
""",
            (source_key, file_hash, mtime, char_count, char_count // 4, chunk_count),
        )
        self._conn.commit()

    def get_file(self, source_key: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM files WHERE source_key = ?", (source_key,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_files(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM files").fetchall()
        return {row["source_key"]: dict(row) for row in rows}

    def get_known_files(self) -> dict[str, str]:
        """Return {source_key: file_hash} for scanner change-detection."""
        rows = self._conn.execute(
            "SELECT source_key, file_hash FROM files WHERE source_missing = 0"
        ).fetchall()
        return {row["source_key"]: row["file_hash"] for row in rows}

    def remove_file(self, source_key: str) -> None:
        self._conn.execute("DELETE FROM files WHERE source_key = ?", (source_key,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Contextual tracking
    # ------------------------------------------------------------------

    def mark_contextual_done(
        self,
        source_key: str,
        backend: str,
        tokens_used: int,
        cost_usd: float,
        used_sliding_window: bool,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """\
UPDATE files SET
    contextual_done     = 1,
    contextual_at       = ?,
    contextual_backend  = ?,
    context_tokens_used = ?,
    context_cost_usd    = ?,
    used_sliding_window = ?
WHERE source_key = ?
""",
            (now, backend, tokens_used, cost_usd, int(used_sliding_window), source_key),
        )
        self._conn.commit()

    def reset_contextual(self, source_key: str) -> None:
        """Clear contextual state for a file (e.g. after re-ingest)."""
        self._conn.execute(
            """\
UPDATE files SET
    contextual_done = 0, contextual_at = NULL,
    contextual_backend = NULL, context_tokens_used = 0,
    context_cost_usd = 0.0, used_sliding_window = 0
WHERE source_key = ?
""",
            (source_key,),
        )
        self._conn.commit()

    def mark_source_missing(self, source_key: str, missing: bool = True) -> None:
        self._conn.execute(
            "UPDATE files SET source_missing = ? WHERE source_key = ?",
            (int(missing), source_key),
        )
        self._conn.commit()

    def get_uncontext_files(self) -> list[dict[str, Any]]:
        """Files not yet contextualized and not missing from disk."""
        rows = self._conn.execute(
            "SELECT * FROM files WHERE contextual_done = 0 AND source_missing = 0"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Ollama benchmarks
    # ------------------------------------------------------------------

    def upsert_benchmark(
        self,
        model: str,
        tok_per_sec: float,
        sample_chunks: int,
        machine_id: str | None = None,
    ) -> None:
        mid = machine_id or _machine_id()
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """\
INSERT INTO ollama_benchmarks (model, machine_id, tok_per_sec, sample_chunks, measured_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(model, machine_id) DO UPDATE SET
    tok_per_sec   = excluded.tok_per_sec,
    sample_chunks = excluded.sample_chunks,
    measured_at   = excluded.measured_at
""",
            (model, mid, tok_per_sec, sample_chunks, now),
        )
        self._conn.commit()

    def get_benchmark(
        self, model: str, machine_id: str | None = None
    ) -> dict[str, Any] | None:
        mid = machine_id or _machine_id()
        row = self._conn.execute(
            "SELECT * FROM ollama_benchmarks WHERE model = ? AND machine_id = ?",
            (model, mid),
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Reset / lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop all tables and recreate.  Used by ``--reset``."""
        self._conn.execute("DROP TABLE IF EXISTS files")
        self._conn.execute("DROP TABLE IF EXISTS ollama_benchmarks")
        self._conn.commit()
        self._migrate()

    def close(self) -> None:
        self._conn.close()
