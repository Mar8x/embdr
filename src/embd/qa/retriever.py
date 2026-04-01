"""
Semantic retriever: embed a query and return the top-k relevant chunks.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..embedding.encoder import Encoder
from ..store.vector_store import K_DOC_DATE, K_DOC_DATE_Q, VectorStore


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

    def __init__(self, store: VectorStore, encoder: Encoder, top_k: int = 5) -> None:
        self._store = store
        self._encoder = encoder
        self._top_k = top_k

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Embed the query and return the top_k most semantically similar chunks."""
        embedding = self._encoder.encode_query(query)
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
