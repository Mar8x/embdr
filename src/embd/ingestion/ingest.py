"""Reusable single-file ingest logic shared by CLI, serve watcher, and shell."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..embedding.encoder import Encoder
from ..perf import Timer
from ..store.vector_store import (
    K_CHUNK_IDX, K_CHUNK_OVL, K_CHUNK_SIZE,
    K_DOC_DATE, K_DOC_DATE_Q,
    K_HASH, K_INGESTED, K_MODEL, K_PAGE, K_SOURCE,
    VectorStore,
)
from .chunker import chunk_pages
from .doc_date import extract_document_date
from .registry import extract_file
from .scanner import hash_file

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    filename: str
    n_chunks: int
    char_count: int
    extract_s: float
    embed_s: float
    upsert_s: float


def _source_key(path: Path, documents_dir: Path) -> str:
    """Derive the source key for a file: relative POSIX path from documents_dir, or bare name."""
    try:
        from pathlib import PurePosixPath
        rel = PurePosixPath(path.relative_to(documents_dir))
        return str(rel)
    except ValueError:
        return path.name


def ingest_file(
    path: Path,
    store: VectorStore,
    encoder: Encoder,
    cfg: Config,
    *,
    delete_existing: bool = False,
    source_name: str | None = None,
) -> IngestResult | None:
    """Ingest a single file into the vector store.

    *source_name* is the key used in the vector store (relative path from
    documents_dir).  When None it is derived automatically.

    Returns None when the file has no extractable text.
    When *delete_existing* is True, removes old chunks for this source first.
    """
    if source_name is None:
        source_name = _source_key(path, cfg.paths.documents_dir)
    logger.info("Ingesting: %s", source_name)

    if delete_existing:
        store.delete_file(source_name)

    file_hash = hash_file(path)
    doc_date = extract_document_date(path)

    with Timer("extract") as t_extract:
        pages = extract_file(path, cfg.ingestion)

    if not pages:
        logger.warning("Skipping '%s': no extractable text", source_name)
        return None

    chunks = chunk_pages(
        pages,
        source_filename=source_name,
        chunk_size=cfg.ingestion.chunk_size,
        chunk_overlap=cfg.ingestion.chunk_overlap,
    )

    texts = [c.text for c in chunks]

    with Timer("embed") as t_embed:
        embeddings = encoder.encode(texts)

    now = datetime.now(timezone.utc).isoformat()
    metadatas = [
        {
            K_SOURCE:     c.source_filename,
            K_PAGE:       c.page_num,
            K_CHUNK_IDX:  c.chunk_index,
            K_INGESTED:   now,
            K_HASH:       file_hash,
            K_MODEL:      encoder.model_version,
            K_CHUNK_SIZE: cfg.ingestion.chunk_size,
            K_CHUNK_OVL:  cfg.ingestion.chunk_overlap,
            K_DOC_DATE:   doc_date.date or "",
            K_DOC_DATE_Q: doc_date.quality,
        }
        for c in chunks
    ]

    with Timer("upsert") as t_upsert:
        store.upsert_chunks(
            chunk_ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas,
        )

    logger.info("Indexed %d chunks from '%s'", len(chunks), source_name)
    return IngestResult(
        filename=source_name,
        n_chunks=len(chunks),
        char_count=sum(len(t) for t in texts),
        extract_s=t_extract.elapsed,
        embed_s=t_embed.elapsed,
        upsert_s=t_upsert.elapsed,
    )
