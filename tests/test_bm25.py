"""Tests for BM25 index build, persist, and query."""
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from embd.ingestion.bm25_index import BM25Index, _BM25State, _tokenize


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []


class TestBuildFromStore:
    def _mock_store(self, docs: list[tuple[str, str]]):
        """Create a mock VectorStore with given (id, text) pairs."""
        mock = MagicMock()
        # Simulate paginated get() — return all at once, then empty
        call_count = [0]
        def fake_get(include, limit, offset):
            if call_count[0] > 0 or offset >= len(docs):
                return {"ids": [], "documents": []}
            call_count[0] += 1
            return {
                "ids": [d[0] for d in docs],
                "documents": [d[1] for d in docs],
            }
        mock._collection.get = fake_get
        return mock

    def test_build_basic(self):
        store = self._mock_store([
            ("c1", "python programming language"),
            ("c2", "java enterprise framework"),
            ("c3", "python data science"),
        ])
        idx = BM25Index.build_from_store(store)
        results = idx.query("python", top_k=3)
        # Both python docs should rank above java
        ids = [r[0] for r in results]
        assert "c1" in ids
        assert "c3" in ids

    def test_build_empty_store(self):
        store = self._mock_store([])
        idx = BM25Index.build_from_store(store)
        results = idx.query("anything", top_k=5)
        assert results == []


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        # Need 3+ docs for BM25 IDF to produce non-zero scores
        docs = [
            ("c1", "hello world greeting"),
            ("c2", "foo bar baz"),
            ("c3", "another document entirely"),
        ]
        store = MagicMock()
        store._collection.get = lambda **kw: (
            {"ids": [d[0] for d in docs], "documents": [d[1] for d in docs]}
            if kw.get("offset", 0) == 0 else {"ids": [], "documents": []}
        )
        idx = BM25Index.build_from_store(store)
        path = tmp_path / "bm25.pkl"
        idx.save(path)

        assert path.exists()
        assert not (tmp_path / "bm25.pkl.tmp").exists()  # tmp cleaned up

        loaded = BM25Index.load(path)
        assert loaded is not None
        results = loaded.query("hello", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "c1"

    def test_load_missing_returns_none(self, tmp_path):
        assert BM25Index.load(tmp_path / "nonexistent.pkl") is None

    def test_load_corrupt_returns_none(self, tmp_path):
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not a pickle")
        assert BM25Index.load(path) is None

    def test_load_wrong_type_returns_none(self, tmp_path):
        path = tmp_path / "wrong.pkl"
        with open(path, "wb") as f:
            pickle.dump({"not": "a BM25State"}, f)
        assert BM25Index.load(path) is None

    def test_atomic_write(self, tmp_path):
        """Verify atomic rename pattern — no .tmp file left behind."""
        store = MagicMock()
        store._collection.get = lambda **kw: (
            {"ids": ["c1"], "documents": ["test doc"]}
            if kw.get("offset", 0) == 0 else {"ids": [], "documents": []}
        )
        idx = BM25Index.build_from_store(store)
        path = tmp_path / "bm25.pkl"
        idx.save(path)

        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "bm25.pkl"


class TestQuery:
    def _build_index(self, docs):
        store = MagicMock()
        store._collection.get = lambda **kw: (
            {"ids": [d[0] for d in docs], "documents": [d[1] for d in docs]}
            if kw.get("offset", 0) == 0 else {"ids": [], "documents": []}
        )
        return BM25Index.build_from_store(store)

    def test_ranking_order(self):
        idx = self._build_index([
            ("c1", "the cat sat on the mat"),
            ("c2", "cat cat cat cat cat"),
            ("c3", "the dog ran in the park"),
        ])
        results = idx.query("cat", top_k=3)
        # c2 has more "cat" occurrences, should rank first
        assert results[0][0] == "c2"

    def test_empty_query(self):
        idx = self._build_index([("c1", "hello world")])
        assert idx.query("", top_k=5) == []

    def test_no_matches(self):
        idx = self._build_index([("c1", "hello world")])
        results = idx.query("xyznonexistent", top_k=5)
        assert results == []

    def test_top_k_limit(self):
        idx = self._build_index([
            (f"c{i}", f"word{i} common term") for i in range(20)
        ])
        results = idx.query("common", top_k=3)
        assert len(results) <= 3
