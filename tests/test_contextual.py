"""Tests for contextual estimation and pricing math."""
import pytest

from embd.ingestion.contextual import (
    CONTEXT_OUTPUT_TOKENS,
    HAIKU_CACHE_READ_MULT,
    HAIKU_CACHE_WRITE_MULT,
    HAIKU_INPUT_PER_M,
    HAIKU_OUTPUT_PER_M,
    OLLAMA_DEFAULT_TOK_S,
    EstimateResult,
    FileEstimate,
    estimate_contextualization,
)
from embd.store.meta_db import MetaDB


@pytest.fixture
def meta_db(tmp_path):
    db = MetaDB(tmp_path / "test.db")
    yield db
    db.close()


class _FakeCfg:
    """Minimal config substitute for estimation tests."""
    class ingestion:
        contextual_backend = "claude"
        contextual_max_doc_tokens = 50000
        contextual_window_chunks = 3
    class llm:
        ollama_model = "llama3.2"


class TestClaudeEstimation:
    def test_full_doc_cost(self, meta_db):
        """Full-doc strategy: 1 cache write + (N-1) reads + N outputs."""
        meta_db.upsert_file("report.pdf", "h1", 100.0, 40000, 20)
        # 40000 chars → 10000 tokens, 20 chunks
        # Token count < 50000 threshold → full doc

        cfg = _FakeCfg()
        result = estimate_contextualization(meta_db, cfg)

        assert len(result.files) == 1
        f = result.files[0]
        assert f.source_key == "report.pdf"
        assert f.token_count == 10000
        assert f.chunk_count == 20
        assert not f.sliding

        # Manual cost calculation
        cache_write = 10000 * HAIKU_CACHE_WRITE_MULT * HAIKU_INPUT_PER_M / 1_000_000
        cache_reads = 10000 * HAIKU_CACHE_READ_MULT * HAIKU_INPUT_PER_M / 1_000_000 * 19
        output_cost = 20 * CONTEXT_OUTPUT_TOKENS * HAIKU_OUTPUT_PER_M / 1_000_000
        expected = cache_write + cache_reads + output_cost

        assert f.cost_usd == pytest.approx(expected, rel=1e-4)

    def test_sliding_window_cost(self, meta_db):
        """Sliding window: when token_count > max_doc_tokens."""
        # 400000 chars → 100000 tokens > 50000 threshold
        meta_db.upsert_file("big.epub", "h2", 200.0, 400000, 200)

        cfg = _FakeCfg()
        result = estimate_contextualization(meta_db, cfg)

        f = result.files[0]
        assert f.sliding
        assert f.token_count == 100000
        assert f.chunk_count == 200

        # Manual: window_tokens = 3 * 2 * (100000/200) = 3000 per chunk
        avg_chunk_tokens = 100000 / 200
        window_tokens = 3 * 2 * avg_chunk_tokens  # 3000
        input_cost = window_tokens * HAIKU_INPUT_PER_M / 1_000_000 * 200
        output_cost = 200 * CONTEXT_OUTPUT_TOKENS * HAIKU_OUTPUT_PER_M / 1_000_000
        expected = input_cost + output_cost
        assert f.cost_usd == pytest.approx(expected, rel=1e-4)

    def test_skips_already_done(self, meta_db):
        meta_db.upsert_file("done.pdf", "h1", 100.0, 5000, 10)
        meta_db.mark_contextual_done("done.pdf", "claude", 1000, 0.01, False)
        meta_db.upsert_file("todo.pdf", "h2", 200.0, 3000, 8)

        cfg = _FakeCfg()
        result = estimate_contextualization(meta_db, cfg)

        assert result.already_done == 1
        assert len(result.files) == 1
        assert result.files[0].source_key == "todo.pdf"

    def test_skips_source_missing(self, meta_db):
        meta_db.upsert_file("gone.pdf", "h1", 100.0, 5000, 10)
        meta_db.mark_source_missing("gone.pdf", True)

        cfg = _FakeCfg()
        result = estimate_contextualization(meta_db, cfg)
        assert len(result.files) == 0


class TestOllamaEstimation:
    def test_time_with_default_throughput(self, meta_db):
        meta_db.upsert_file("doc.pdf", "h1", 100.0, 8000, 20)

        cfg = _FakeCfg()
        cfg.ingestion.contextual_backend = "ollama"
        result = estimate_contextualization(meta_db, cfg)

        f = result.files[0]
        total_output = 20 * CONTEXT_OUTPUT_TOKENS
        expected_seconds = total_output / OLLAMA_DEFAULT_TOK_S
        assert f.time_seconds == pytest.approx(expected_seconds, rel=1e-4)
        assert not result.tok_per_sec_measured

    def test_time_with_benchmark(self, meta_db):
        meta_db.upsert_file("doc.pdf", "h1", 100.0, 8000, 20)
        meta_db.upsert_benchmark("llama3.2", 60.0, 100)

        cfg = _FakeCfg()
        cfg.ingestion.contextual_backend = "ollama"
        result = estimate_contextualization(meta_db, cfg)

        f = result.files[0]
        total_output = 20 * CONTEXT_OUTPUT_TOKENS
        expected_seconds = total_output / 60.0
        assert f.time_seconds == pytest.approx(expected_seconds, rel=1e-4)
        assert result.tok_per_sec_measured


class TestEmptyEstimate:
    def test_no_files(self, meta_db):
        cfg = _FakeCfg()
        result = estimate_contextualization(meta_db, cfg)
        assert len(result.files) == 0
        assert result.already_done == 0
