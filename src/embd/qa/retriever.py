"""
Semantic retriever: embed a query and return the top-k relevant chunks.

Supports optional BM25 hybrid search via Reciprocal Rank Fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..embedding.encoder import Encoder
from ..store.vector_store import K_DOC_DATE, K_DOC_DATE_Q, VectorStore

if TYPE_CHECKING:
    from ..ingestion.bm25_index import BM25Index


@dataclass
class RetrievedChunk:
    text: str
    source_filename: str
    page_num: int
    chunk_index: int
    distance: float          # cosine distance; lower = more similar (0 = identical)
    document_date: str = ""  # ISO-8601 (best-effort); empty when unknown
    document_date_quality: str = "none"  # "document_metadata" | "filesystem" | "none"


class Retriever:
    """Combines encoding and vector search into a single retrieve() call.

    Also available for direct use by application code that wants to bypass
    the CLI. The CLI query command calls encode_query + store.query directly
    so it can time each step independently.
    """

    def __init__(
        self,
        store: VectorStore,
        encoder: Encoder,
        top_k: int = 5,
        bm25_index: BM25Index | None = None,
        semantic_weight: float = 0.8,
        bm25_weight: float = 0.2,
    ) -> None:
        self._store = store
        self._encoder = encoder
        self._top_k = top_k
        self._bm25 = bm25_index
        self._semantic_weight = semantic_weight
        self._bm25_weight = bm25_weight

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Embed the query and return the top_k most relevant chunks.

        When a BM25 index is available, results are merged via Reciprocal
        Rank Fusion (semantic + keyword).  Falls back to pure semantic
        when no BM25 index is loaded.
        """
        embedding = self._encoder.encode_query(query)

        if self._bm25 is not None and self._bm25_weight > 0:
            from .hybrid_retriever import rrf_merge

            # Over-fetch both sources so RRF has enough candidates
            fetch_k = self._top_k * 2
            semantic_hits = self._store.query(embedding, top_k=fetch_k)
            bm25_hits = self._bm25.query(query, top_k=fetch_k)
            hits = rrf_merge(
                semantic_hits, bm25_hits, self._store,
                semantic_weight=self._semantic_weight,
                bm25_weight=self._bm25_weight,
                top_k=self._top_k,
            )
        else:
            hits = self._store.query(embedding, top_k=self._top_k)

        return [
            RetrievedChunk(
                text=h["text"],
                source_filename=h["metadata"]["source_filename"],
                page_num=int(h["metadata"]["page_num"]),
                chunk_index=int(h["metadata"]["chunk_index"]),
                distance=h["distance"],
                document_date=h["metadata"].get(K_DOC_DATE, ""),
                document_date_quality=h["metadata"].get(K_DOC_DATE_Q, "none"),
            )
            for h in hits
        ]
