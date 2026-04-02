"""Tests for the SQLite metadata database (MetaDB)."""
import sqlite3
from pathlib import Path

import pytest

from embd.store.meta_db import MetaDB, _machine_id


@pytest.fixture
def db(tmp_path):
    meta = MetaDB(tmp_path / "test_meta.db")
    yield meta
    meta.close()


class TestMigration:
    def test_creates_tables(self, db):
        """Tables exist after init."""
        row = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='files'"
        ).fetchone()
        assert row is not None

        row = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ollama_benchmarks'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self, tmp_path):
        """Running migrate twice does not raise."""
        path = tmp_path / "idempotent.db"
        db1 = MetaDB(path)
        db1.close()
        db2 = MetaDB(path)  # second open triggers _migrate again
        db2.close()

    def test_adds_columns_to_existing_db(self, tmp_path):
        """Columns are added to an existing table missing them."""
        path = tmp_path / "old.db"
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE files (source_key TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE ollama_benchmarks "
            "(model TEXT, machine_id TEXT, PRIMARY KEY (model, machine_id))"
        )
        conn.commit()
        conn.close()

        db = MetaDB(path)
        # Should have all columns now
        db.upsert_file("test.pdf", "abc123", 1000.0, 5000, 10)
        row = db.get_file("test.pdf")
        assert row is not None
        assert row["token_count"] == 1250  # 5000 // 4
        db.close()


class TestFileCRUD:
    def test_upsert_and_get(self, db):
        db.upsert_file("report.pdf", "hash1", 1000.0, 8000, 20)
        row = db.get_file("report.pdf")
        assert row["file_hash"] == "hash1"
        assert row["char_count"] == 8000
        assert row["token_count"] == 2000
        assert row["chunk_count"] == 20
        assert row["contextual_done"] == 0

    def test_upsert_overwrites(self, db):
        db.upsert_file("report.pdf", "hash1", 1000.0, 8000, 20)
        db.upsert_file("report.pdf", "hash2", 2000.0, 9000, 25)
        row = db.get_file("report.pdf")
        assert row["file_hash"] == "hash2"
        assert row["chunk_count"] == 25

    def test_get_missing_returns_none(self, db):
        assert db.get_file("nonexistent.pdf") is None

    def test_get_all_files(self, db):
        db.upsert_file("a.pdf", "h1", 100.0, 1000, 5)
        db.upsert_file("b.pdf", "h2", 200.0, 2000, 10)
        all_f = db.get_all_files()
        assert len(all_f) == 2
        assert "a.pdf" in all_f
        assert "b.pdf" in all_f

    def test_get_known_files(self, db):
        db.upsert_file("a.pdf", "h1", 100.0, 1000, 5)
        db.upsert_file("b.pdf", "h2", 200.0, 2000, 10)
        db.mark_source_missing("b.pdf", True)
        known = db.get_known_files()
        assert known == {"a.pdf": "h1"}

    def test_remove_file(self, db):
        db.upsert_file("a.pdf", "h1", 100.0, 1000, 5)
        db.remove_file("a.pdf")
        assert db.get_file("a.pdf") is None


class TestContextualTracking:
    def test_mark_done(self, db):
        db.upsert_file("doc.pdf", "h1", 100.0, 5000, 12)
        db.mark_contextual_done("doc.pdf", "claude", 3000, 0.05, False)
        row = db.get_file("doc.pdf")
        assert row["contextual_done"] == 1
        assert row["contextual_backend"] == "claude"
        assert row["context_tokens_used"] == 3000
        assert abs(row["context_cost_usd"] - 0.05) < 1e-6
        assert row["contextual_at"] is not None

    def test_reset_contextual(self, db):
        db.upsert_file("doc.pdf", "h1", 100.0, 5000, 12)
        db.mark_contextual_done("doc.pdf", "claude", 3000, 0.05, False)
        db.reset_contextual("doc.pdf")
        row = db.get_file("doc.pdf")
        assert row["contextual_done"] == 0
        assert row["context_tokens_used"] == 0

    def test_get_uncontext_files(self, db):
        db.upsert_file("done.pdf", "h1", 100.0, 5000, 12)
        db.mark_contextual_done("done.pdf", "claude", 3000, 0.05, False)
        db.upsert_file("todo.pdf", "h2", 200.0, 3000, 8)
        db.upsert_file("missing.pdf", "h3", 300.0, 1000, 3)
        db.mark_source_missing("missing.pdf", True)

        uncontext = db.get_uncontext_files()
        assert len(uncontext) == 1
        assert uncontext[0]["source_key"] == "todo.pdf"


class TestBenchmarks:
    def test_upsert_and_get(self, db):
        db.upsert_benchmark("llama3.2", 52.0, 100, machine_id="test123")
        row = db.get_benchmark("llama3.2", machine_id="test123")
        assert row is not None
        assert abs(row["tok_per_sec"] - 52.0) < 0.01
        assert row["sample_chunks"] == 100

    def test_upsert_updates(self, db):
        db.upsert_benchmark("llama3.2", 50.0, 100, machine_id="test123")
        db.upsert_benchmark("llama3.2", 55.0, 200, machine_id="test123")
        row = db.get_benchmark("llama3.2", machine_id="test123")
        assert abs(row["tok_per_sec"] - 55.0) < 0.01
        assert row["sample_chunks"] == 200

    def test_missing_returns_none(self, db):
        assert db.get_benchmark("nonexistent") is None


class TestReset:
    def test_reset_clears_all(self, db):
        db.upsert_file("a.pdf", "h1", 100.0, 1000, 5)
        db.upsert_benchmark("model", 50.0, 10, machine_id="m1")
        db.reset()
        assert db.get_all_files() == {}
        assert db.get_benchmark("model", machine_id="m1") is None


class TestMachineId:
    def test_deterministic(self):
        assert _machine_id() == _machine_id()

    def test_length(self):
        assert len(_machine_id()) == 12
