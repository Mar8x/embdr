"""
ChromaDB-backed vector store for document chunks.

Uses a single collection for all documents. Per-file operations
(delete, change-detection) use metadata filtering which ChromaDB
handles efficiently through its SQLite metadata layer.

All metadata field names are defined as module-level constants so
other modules import the constants rather than repeating string literals.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Metadata key constants — import these from other modules to avoid typos
K_SOURCE     = "source_filename"
K_PAGE       = "page_num"
K_CHUNK_IDX  = "chunk_index"
K_INGESTED   = "ingestion_timestamp"
K_HASH       = "file_hash"
K_MODEL      = "embedding_model"
K_CHUNK_SIZE = "chunk_size"
K_CHUNK_OVL  = "chunk_overlap"
K_DOC_DATE   = "document_date"       # ISO-8601 date string (best-effort)
K_DOC_DATE_Q = "document_date_quality"  # "document_metadata" | "filesystem" | "none"


class VectorStore:
    """Manages all interactions with the local ChromaDB collection.

    Design choices:
    - Single collection: all documents coexist; cross-document queries work naturally.
    - hnsw:space=cosine: matches L2-normalized sentence-transformer embeddings.
      Changing this after data is stored requires a full rebuild.
    - anonymized_telemetry=False: no outbound network calls from ChromaDB.
    - Batch upsert in groups of 500 to stay within ChromaDB's default limits.
    """

    def __init__(self, db_dir: Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(
            "VectorStore ready: %s  collection=%s  chunks=%d",
            db_dir, collection_name, self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert in batches of 500 to stay within ChromaDB limits.

        ChromaDB upsert is idempotent: if a chunk_id already exists with
        identical data, it is effectively a no-op.
        """
        batch_size = 500
        for i in range(0, len(chunk_ids), batch_size):
            s = slice(i, i + batch_size)
            self._collection.upsert(
                ids=chunk_ids[s],
                embeddings=embeddings[s],
                documents=texts[s],
                metadatas=metadatas[s],
            )
        logger.info("Upserted %d chunks into collection", len(chunk_ids))

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_file(self, filename: str) -> int:
        """Delete all chunks that belong to filename. Returns count deleted.

        Uses include=[] to fetch IDs only — no embeddings or document text
        is transferred, keeping this operation lightweight even for files
        with thousands of chunks.
        """
        results = self._collection.get(
            where={K_SOURCE: filename},
            include=[],
        )
        ids: list[str] = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Deleted %d chunks for '%s'", len(ids), filename)
        else:
            logger.debug("No chunks found for '%s' — nothing deleted", filename)
        return len(ids)

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def get_known_files(self) -> dict[str, str]:
        """Return {filename: file_hash} for every file currently indexed.

        Paginates in batches of 500 to avoid SQLite's bound-variable limit.
        An unbounded get() on a large collection exceeds SQLite's default
        999-variable cap and raises an InternalError.
        """
        seen: dict[str, str] = {}
        offset = 0
        batch = 500

        while True:
            results = self._collection.get(
                include=["metadatas"],
                limit=batch,
                offset=offset,
            )
            metadatas = results["metadatas"]
            for meta in metadatas:
                fname = meta.get(K_SOURCE, "")
                fhash = meta.get(K_HASH, "")
                if fname and fname not in seen:
                    seen[fname] = fhash
            if len(metadatas) < batch:
                break
            offset += batch

        return seen

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top_k most relevant chunks for a query embedding.

        Each result dict contains:
            id (str), text (str), metadata (dict), distance (float)

        Results are sorted by distance ascending (most relevant first).
        The optional `where` filter restricts results to a specific file.
        """
        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        return [
            {
                "id":       results["ids"][0][i],
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()
