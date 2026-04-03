"""Tests for Reciprocal Rank Fusion merge."""
from unittest.mock import MagicMock

import pytest

from embd.qa.hybrid_retriever import rrf_merge, MergeResult


def _make_hit(cid, distance=0.5, text="text", metadata=None):
    return {
        "id": cid,
        "distance": distance,
        "text": text,
        "metadata": metadata or {"source_filename": "test.pdf", "page_num": 1, "chunk_index": 0},
    }


def _mock_store(extra_chunks=None):
    """Store that can fetch missing chunks by ID."""
    extra = extra_chunks or {}
    mock = MagicMock()
    def fake_get(ids, include):
        return {
            "ids": [i for i in ids if i in extra],
            "documents": [extra[i]["text"] for i in ids if i in extra],
            "metadatas": [extra[i]["metadata"] for i in ids if i in extra],
        }
    mock._collection.get = fake_get
    return mock


class TestRRFMerge:
    def test_pure_semantic(self):
        """When BM25 returns nothing, result is semantic order."""
        semantic = [_make_hit("c1"), _make_hit("c2"), _make_hit("c3")]
        merge = rrf_merge(semantic, [], _mock_store(), top_k=3)
        assert isinstance(merge, MergeResult)
        assert [r["id"] for r in merge.results] == ["c1", "c2", "c3"]

    def test_pure_bm25(self):
        """When semantic returns nothing, BM25 results are fetched from store."""
        extra = {
            "b1": {"text": "bm25 text 1", "metadata": {"source_filename": "a.pdf", "page_num": 1, "chunk_index": 0}},
            "b2": {"text": "bm25 text 2", "metadata": {"source_filename": "a.pdf", "page_num": 1, "chunk_index": 1}},
        }
        bm25 = [("b1", 5.0), ("b2", 3.0)]
        merge = rrf_merge([], bm25, _mock_store(extra), top_k=2)
        assert len(merge.results) == 2
        assert merge.results[0]["id"] == "b1"
        # Diagnostics: both should be bm25-only
        assert all(d.source == "bm25" for d in merge.diag)

    def test_overlap_dedup(self):
        """Same chunk from both sources is not duplicated."""
        semantic = [_make_hit("c1"), _make_hit("c2")]
        bm25 = [("c1", 5.0), ("c3", 3.0)]
        extra = {
            "c3": {"text": "t3", "metadata": {"source_filename": "a.pdf", "page_num": 1, "chunk_index": 2}},
        }
        merge = rrf_merge(semantic, bm25, _mock_store(extra), top_k=3)
        ids = [r["id"] for r in merge.results]
        assert ids.count("c1") == 1  # no duplication
        # c1 should be marked as from both sources
        c1_diag = next(d for d in merge.diag if d.chunk_id == "c1")
        assert c1_diag.source == "semantic+bm25"

    def test_bm25_boosts_ranking(self):
        """A chunk ranked low in semantic but high in BM25 gets boosted."""
        semantic = [_make_hit("c1"), _make_hit("c2"), _make_hit("c3")]
        bm25 = [("c3", 10.0), ("c2", 5.0)]  # c3 is BM25 top
        merge = rrf_merge(
            semantic, bm25, _mock_store(),
            semantic_weight=1.0, bm25_weight=1.0, top_k=3,
        )
        ids = [r["id"] for r in merge.results]
        assert ids.index("c3") < 2

    def test_top_k_limits_output(self):
        semantic = [_make_hit(f"c{i}") for i in range(10)]
        merge = rrf_merge(semantic, [], _mock_store(), top_k=3)
        assert len(merge.results) == 3
        assert len(merge.diag) == 3

    def test_distance_field_is_rrf_score(self):
        """Distance should be derived from RRF score, not original cosine."""
        semantic = [_make_hit("c1", distance=0.8)]
        merge = rrf_merge(semantic, [], _mock_store(), top_k=1)
        assert merge.results[0]["distance"] == pytest.approx(0.0, abs=0.01)

    def test_weights_affect_ranking(self):
        """Higher BM25 weight makes BM25-top results rank higher."""
        semantic = [_make_hit("c1"), _make_hit("c2")]
        bm25 = [("c2", 10.0), ("c1", 1.0)]

        merge = rrf_merge(
            semantic, bm25, _mock_store(),
            semantic_weight=0.1, bm25_weight=10.0, top_k=2,
        )
        assert merge.results[0]["id"] == "c2"

    def test_empty_inputs(self):
        merge = rrf_merge([], [], _mock_store(), top_k=5)
        assert merge.results == []
        assert merge.diag == []

    def test_diag_ranks(self):
        """Diagnostics report correct semantic and BM25 ranks."""
        semantic = [_make_hit("c1"), _make_hit("c2")]
        bm25 = [("c2", 5.0), ("c3", 3.0)]
        extra = {
            "c3": {"text": "t3", "metadata": {"source_filename": "a.pdf", "page_num": 1, "chunk_index": 2}},
        }
        merge = rrf_merge(semantic, bm25, _mock_store(extra), top_k=3)

        diag_by_id = {d.chunk_id: d for d in merge.diag}
        # c1: semantic rank 0, no bm25
        assert diag_by_id["c1"].semantic_rank == 0
        assert diag_by_id["c1"].bm25_rank is None
        assert diag_by_id["c1"].source == "semantic"
        # c2: semantic rank 1, bm25 rank 0
        assert diag_by_id["c2"].semantic_rank == 1
        assert diag_by_id["c2"].bm25_rank == 0
        assert diag_by_id["c2"].source == "semantic+bm25"
        # c3: no semantic, bm25 rank 1
        assert diag_by_id["c3"].semantic_rank is None
        assert diag_by_id["c3"].bm25_rank == 1
        assert diag_by_id["c3"].source == "bm25"
