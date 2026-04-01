"""
Performance instrumentation for embd.

Every CLI command uses Timer to measure individual steps, then calls a
print_*_report function to display a consistent summary to the user.
This makes it easy to spot where time is actually going (embedding model
load, HNSW search, LLM generation, etc.).
"""
from __future__ import annotations

import time
from typing import Any

import psutil

from .qa.token_usage import TokenUsage


class Timer:
    """High-resolution context manager timer using time.perf_counter()."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


def _rss_gb() -> float:
    """Current process Resident Set Size in gigabytes.

    On Apple Silicon with unified memory, RSS reflects how much of the
    shared memory pool this process is using, giving a reasonable proxy
    for 'how much memory is the model consuming right now'.
    """
    return psutil.Process().memory_info().rss / (1024 ** 3)


def print_ingest_report(
    *,
    n_new: int,
    n_updated: int,
    n_removed: int,
    n_chunks: int,
    embed_s: float,
    total_s: float,
) -> None:
    """Print a formatted ingest performance summary."""
    rss = _rss_gb()
    chunks_per_s = n_chunks / embed_s if (embed_s > 0 and n_chunks > 0) else 0.0
    width = 50

    print(f"\n── Ingest Performance {'─' * (width - 21)}")
    print(f"  Files:      {n_new} new, {n_updated} updated, {n_removed} removed")
    print(f"  Chunks:     {n_chunks} total")
    if embed_s > 0 and n_chunks > 0:
        print(f"  Embed time: {embed_s:.1f}s  ({chunks_per_s:.1f} chunks/s)")
    print(f"  Total time: {total_s:.1f}s")
    print(f"  Memory RSS: {rss:.2f} GB")
    print(f"{'─' * (width + 2)}\n")


def print_query_report(
    *,
    embed_ms: float,
    retrieve_ms: float,
    generate_s: float,
    usage: TokenUsage | None,
    top_k: int,
) -> None:
    """Print a formatted query performance summary."""
    rss = _rss_gb()
    width = 50
    total_s = embed_ms / 1000 + retrieve_ms / 1000 + generate_s

    out_tok = usage.completion_tokens if usage is not None else None
    tok_info = ""
    if out_tok is not None and generate_s > 0:
        tok_info = f"  ({out_tok / generate_s:.0f} tok/s, {out_tok} out tok)"

    print(f"\n── Query Performance {'─' * (width - 20)}")
    print(f"  Embed query:     {embed_ms:.0f}ms")
    print(f"  Retrieve top-{top_k}: {retrieve_ms:.0f}ms")
    print(f"  Generate:        {generate_s:.1f}s{tok_info}")
    print(f"  Total:           {total_s:.1f}s")
    if usage is not None:
        pi = f"{usage.prompt_tokens:,}" if usage.prompt_tokens is not None else "—"
        co = f"{usage.completion_tokens:,}" if usage.completion_tokens is not None else "—"
        tot = f"{usage.total_tokens:,}" if usage.total_tokens is not None else "—"
        print(f"  LLM tokens:      in={pi}  out={co}  total={tot}")
    print(f"  Memory RSS: {rss:.2f} GB")
    print(f"{'─' * (width + 2)}\n")
