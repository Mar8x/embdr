"""
BM25 keyword index built from all chunks in ChromaDB.

The index is persisted as a pickle alongside ``chroma_db/``.
Atomic writes (tmp + rename) prevent corruption from interrupted saves.
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path relative to db_dir
BM25_FILENAME = "bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    """Lowercase split — simple but effective for BM25."""
    return text.lower().split()


@dataclass
class _BM25State:
    """Serializable BM25 state."""
    bm25: object  # BM25Okapi
    chunk_ids: list[str]


class BM25Index:
    """In-memory BM25 index with persist/load."""

    def __init__(self, state: _BM25State) -> None:
        self._state = state

    @classmethod
    def build_from_store(cls, store: "VectorStore") -> BM25Index:
        """Read all chunks from ChromaDB and build a BM25 index."""
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

        chunk_ids: list[str] = []
        corpus: list[list[str]] = []
        batch = 500
        offset = 0

        while True:
            results = store._collection.get(
                include=["documents"],
                limit=batch,
                offset=offset,
            )
            ids = results["ids"]
            docs = results["documents"]
            if not ids:
                break
            for cid, text in zip(ids, docs):
                chunk_ids.append(cid)
                corpus.append(_tokenize(text or ""))
            if len(ids) < batch:
                break
            offset += batch

        if not corpus:
            # Empty index — still valid, returns no results
            corpus = [["_empty_"]]
            chunk_ids = ["__empty__"]

        bm25 = BM25Okapi(corpus)
        logger.info("Built BM25 index: %d chunks", len(chunk_ids))
        return cls(_BM25State(bm25=bm25, chunk_ids=chunk_ids))

    def save(self, path: Path) -> None:
        """Atomic write: pickle to .tmp then rename."""
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self._state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(str(tmp), str(path))
        logger.info("Saved BM25 index to %s", path)

    @classmethod
    def load(cls, path: Path) -> BM25Index | None:
        """Load from pickle. Returns None if missing or corrupt."""
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)  # noqa: S301
            if not isinstance(state, _BM25State):
                logger.warning("BM25 pickle has unexpected type, ignoring")
                return None
            logger.debug("Loaded BM25 index: %d chunks", len(state.chunk_ids))
            return cls(state)
        except Exception:
            logger.warning("Failed to load BM25 index from %s, ignoring", path, exc_info=True)
            return None

    def query(self, query_text: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return (chunk_id, score) pairs sorted by BM25 score descending."""
        tokens = _tokenize(query_text)
        if not tokens:
            return []
        scores = self._state.bm25.get_scores(tokens)
        # Get top_k indices by score
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            (self._state.chunk_ids[i], float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]


def build_and_save_bm25(store: "VectorStore", db_dir: Path) -> None:
    """Convenience: build from store and persist atomically."""
    idx = BM25Index.build_from_store(store)
    idx.save(db_dir / BM25_FILENAME)
