"""
Reciprocal Rank Fusion (RRF) for merging semantic and BM25 results.

RRF is rank-based, making it robust to the incomparable score scales
of cosine distance (0–2) and BM25 scores (unbounded).

Formula: score(d) = sum( weight / (k + rank + 1) ) across all lists
where k=60 is the standard constant (Cormack et al.).
"""
from __future__ import annotations

from typing import Any


def rrf_merge(
    semantic_results: list[dict[str, Any]],
    bm25_results: list[tuple[str, float]],
    store: Any,  # VectorStore — avoid circular import
    semantic_weight: float = 1.0,
    bm25_weight: float = 1.0,
    k: int = 60,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Merge semantic (vector) and BM25 results via Reciprocal Rank Fusion.

    Returns results in the same format as ``VectorStore.query()``:
    ``[{id, text, metadata, distance}, ...]`` sorted by fused score descending.

    The ``distance`` field is replaced with a synthetic RRF score in [0, 1]
    (higher = more relevant) so downstream code can rank uniformly.
    """
    scores: dict[str, float] = {}

    # Semantic contribution
    for rank, hit in enumerate(semantic_results):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0.0) + semantic_weight / (k + rank + 1)

    # BM25 contribution
    for rank, (cid, _bm25_score) in enumerate(bm25_results):
        scores[cid] = scores.get(cid, 0.0) + bm25_weight / (k + rank + 1)

    # Sort by fused score descending, take top_k
    ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]

    # Build a lookup of already-fetched results from semantic
    sem_by_id: dict[str, dict[str, Any]] = {h["id"]: h for h in semantic_results}

    # Identify IDs that came only from BM25 (need metadata fetch)
    missing_ids = [cid for cid in ranked_ids if cid not in sem_by_id]
    fetched: dict[str, dict[str, Any]] = {}
    if missing_ids:
        results = store._collection.get(
            ids=missing_ids,
            include=["documents", "metadatas"],
        )
        for i, cid in enumerate(results["ids"]):
            fetched[cid] = {
                "id": cid,
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
                "distance": 0.5,  # placeholder — no real distance for BM25-only hits
            }

    # Normalize scores to [0, 1] for the distance field (inverted: lower distance = better)
    max_score = max(scores.values()) if scores else 1.0

    merged: list[dict[str, Any]] = []
    for cid in ranked_ids:
        hit = sem_by_id.get(cid) or fetched.get(cid)
        if hit is None:
            continue
        result = dict(hit)
        # Replace distance with RRF-based distance: 1 - normalized_score
        result["distance"] = 1.0 - (scores[cid] / max_score) if max_score > 0 else 0.5
        merged.append(result)

    return merged
