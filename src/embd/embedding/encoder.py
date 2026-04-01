"""
Sentence-transformer embedding encoder with automatic device selection.

Key decisions:
- device="auto" default: picks MPS (Apple Silicon), CUDA, or CPU automatically.
- normalize_embeddings=True: L2-normalizes vectors so cosine similarity
  can be computed as a plain dot product (ChromaDB hnsw:space=cosine matches).
  BGE-M3 normalizes by default, but we keep this explicit for other models.
- BGE query prefix: BGE-small/base v1.5 models use asymmetric retrieval
  and need a query prefix. BGE-M3 does NOT need any prefix.
- Lazy loading via cached_property: the model loads only on first encode()
  call, so importing this module or running `embd --help` is instant.
- MPS memory management: explicit cache clearing between batches and
  conservative batch sizes prevent MPS OOM on 48 GB unified memory.
"""
from __future__ import annotations

import logging
from functools import cached_property

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_QUERY_PREFIXES: dict[str, str] = {
    "bge-small-en": "Represent this sentence for searching relevant passages: ",
    "bge-base-en": "Represent this sentence for searching relevant passages: ",
    "bge-large-en": "Represent this sentence for searching relevant passages: ",
}


def _get_query_prefix(model_name: str) -> str:
    """Return the query prefix for a model, or empty string if none needed."""
    lower = model_name.lower()
    for key, prefix in _QUERY_PREFIXES.items():
        if key in lower:
            return prefix
    return ""


def _resolve_device(requested: str) -> str:
    """Resolve ``"auto"`` to the best available accelerator."""
    if requested and requested.lower() != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _flush_gpu_cache() -> None:
    """Release unused GPU memory back to the system."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


class Encoder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self._device = _resolve_device(device)
        self._query_prefix = _get_query_prefix(model_name)

    @cached_property
    def _model(self) -> SentenceTransformer:
        logger.info(
            "Loading embedding model '%s' on device=%s …",
            self.model_name,
            self._device or "auto",
        )
        model = SentenceTransformer(self.model_name, device=self._device)
        dim = model.get_sentence_embedding_dimension()
        max_len = model.max_seq_length
        logger.info(
            "Embedding model ready. Dimension: %d, max_seq_length: %d, "
            "sentence-transformers device=%s (use Activity Monitor “GPU” while encoding to see Metal load).",
            dim,
            max_len,
            self._device,
        )
        return model

    def _auto_batch_size(self) -> int:
        """Pick a safe batch size based on model dimension and max sequence length.

        BGE-M3 (dim=1024, max_seq=8192) is extremely memory-hungry per sample.
        A single batch of 16 long texts can request >20 GiB on MPS.
        """
        dim = self._model.get_sentence_embedding_dimension()
        max_len = self._model.max_seq_length
        if dim >= 1024 and max_len >= 4096:
            return 4
        if dim >= 1024:
            return 8
        if dim >= 768:
            return 32
        return 128

    def encode(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Encode passage texts. Returns plain Python float lists for ChromaDB.

        Processes texts in sub-batches with MPS cache clearing between them
        to keep peak memory well within the unified-memory budget.
        """
        if batch_size is None:
            batch_size = self._auto_batch_size()

        all_embeddings: list[np.ndarray] = []
        show_bar = len(texts) > 5

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            emb: np.ndarray = self._model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            all_embeddings.append(emb)
            _flush_gpu_cache()

            if show_bar:
                done = min(start + batch_size, len(texts))
                logger.info("Encoded %d / %d texts", done, len(texts))

        result = np.vstack(all_embeddings)
        return result.tolist()

    def encode_query(self, query: str) -> list[float]:
        """Encode a single query string, applying model-specific prefix if needed."""
        prefixed = self._query_prefix + query if self._query_prefix else query
        return self.encode([prefixed], batch_size=1)[0]

    @property
    def model_version(self) -> str:
        """Stored in chunk metadata to detect model mismatches after a model swap."""
        return self.model_name
