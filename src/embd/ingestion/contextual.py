"""
Contextual generation: prepend LLM-generated context to each chunk.

Each chunk is sent to an LLM alongside the full document text (or a
sliding window for large documents).  The LLM returns a short ~75-token
context string that situates the chunk within the document.  The chunk
text is then replaced with ``context + "\n\n" + original_text``, re-embedded,
and updated in ChromaDB.

Supports two backends:
  - **claude** — fast, cheap with prompt caching (cache_control: ephemeral).
  - **ollama** — local, free, no API cost, slower.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import click

from ..config import Config
from ..embedding.encoder import Encoder
from ..perf import Timer
from ..store.meta_db import MetaDB
from ..store.vector_store import K_SOURCE, VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing constants (USD per million tokens)
# ---------------------------------------------------------------------------

HAIKU_INPUT_PER_M = 0.80
HAIKU_OUTPUT_PER_M = 4.00
HAIKU_CACHE_WRITE_MULT = 1.25   # 5-minute TTL
HAIKU_CACHE_READ_MULT = 0.10
CONTEXT_OUTPUT_TOKENS = 75
OLLAMA_DEFAULT_TOK_S = 40.0

CONTEXTUAL_CLAUDE_MODEL = "claude-haiku-4-5-20251001"

_CONTEXT_PROMPT = """\
<document>
{doc_text}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

_CONTEXT_PROMPT_WINDOW = """\
<surrounding_context>
{window_text}
</surrounding_context>

Here is the chunk we want to situate within its surrounding context:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

@dataclass
class FileEstimate:
    source_key: str
    chunk_count: int
    char_count: int
    token_count: int
    sliding: bool
    cost_usd: float = 0.0
    time_seconds: float = 0.0


@dataclass
class EstimateResult:
    backend: str
    files: list[FileEstimate] = field(default_factory=list)
    already_done: int = 0
    tok_per_sec: float = OLLAMA_DEFAULT_TOK_S
    tok_per_sec_measured: bool = False


def estimate_contextualization(meta_db: MetaDB, cfg: Config) -> EstimateResult:
    """Estimate cost/time for contextual generation.  Reads SQLite only."""
    backend = cfg.ingestion.contextual_backend
    max_doc_tokens = cfg.ingestion.contextual_max_doc_tokens
    window_chunks = cfg.ingestion.contextual_window_chunks

    all_files = meta_db.get_all_files()
    uncontext = [f for f in all_files.values() if not f["contextual_done"] and not f["source_missing"]]
    already_done = sum(1 for f in all_files.values() if f["contextual_done"])

    result = EstimateResult(backend=backend, already_done=already_done)

    # Ollama throughput lookup
    if backend == "ollama":
        bench = meta_db.get_benchmark(cfg.llm.ollama_model)
        if bench:
            result.tok_per_sec = bench["tok_per_sec"]
            result.tok_per_sec_measured = True

    for f in uncontext:
        token_count = f["token_count"] or 0
        chunk_count = f["chunk_count"] or 1
        char_count = f["char_count"] or 0
        sliding = token_count > max_doc_tokens

        est = FileEstimate(
            source_key=f["source_key"],
            chunk_count=chunk_count,
            char_count=char_count,
            token_count=token_count,
            sliding=sliding,
        )

        if backend == "claude":
            if not sliding:
                # Full doc cached: 1 cache write + (N-1) cache reads + N outputs
                cache_write = token_count * HAIKU_CACHE_WRITE_MULT * HAIKU_INPUT_PER_M / 1_000_000
                cache_reads = token_count * HAIKU_CACHE_READ_MULT * HAIKU_INPUT_PER_M / 1_000_000 * max(chunk_count - 1, 0)
                output_cost = chunk_count * CONTEXT_OUTPUT_TOKENS * HAIKU_OUTPUT_PER_M / 1_000_000
                est.cost_usd = cache_write + cache_reads + output_cost
            else:
                avg_chunk_tokens = token_count / chunk_count if chunk_count else 0
                window_tokens = window_chunks * 2 * avg_chunk_tokens
                input_cost = window_tokens * HAIKU_INPUT_PER_M / 1_000_000 * chunk_count
                output_cost = chunk_count * CONTEXT_OUTPUT_TOKENS * HAIKU_OUTPUT_PER_M / 1_000_000
                est.cost_usd = input_cost + output_cost
        elif backend == "ollama":
            total_output = chunk_count * CONTEXT_OUTPUT_TOKENS
            est.time_seconds = total_output / result.tok_per_sec

        result.files.append(est)

    return result


def print_estimate(estimate: EstimateResult, cfg: Config) -> None:
    """Print a formatted cost/time estimate table."""
    if not estimate.files:
        click.echo("All files are already contextualized — nothing to do.")
        return

    is_claude = estimate.backend == "claude"
    header_last = "est. cost" if is_claude else "est. time"

    click.echo(f"\n  {'file':<40s} {'chunks':>6s}   {'chars':>10s}   {'tokens':>8s}   {'window':<10s} {header_last:>10s}")
    click.echo("  " + "─" * 90)

    total_cost = 0.0
    total_time = 0.0
    total_chunks = 0
    total_chars = 0
    total_tokens = 0

    for f in estimate.files:
        window_str = "sliding" if f.sliding else "full doc"
        if is_claude:
            val_str = f"${f.cost_usd:.2f}"
            total_cost += f.cost_usd
        else:
            val_str = _fmt_seconds(f.time_seconds)
            total_time += f.time_seconds
        total_chunks += f.chunk_count
        total_chars += f.char_count
        total_tokens += f.token_count
        click.echo(
            f"  {f.source_key:<40s} {f.chunk_count:>6d}   {f.char_count:>10,d}   "
            f"{f.token_count:>8,d}   {window_str:<10s} {val_str:>10s}"
        )

    click.echo("  " + "─" * 90)
    total_val = f"${total_cost:.2f}" if is_claude else _fmt_seconds(total_time)
    click.echo(
        f"  {'total':<40s} {total_chunks:>6d}   {total_chars:>10,d}   "
        f"{total_tokens:>8,d}   {'':<10s} {total_val:>10s}"
    )

    click.echo()
    if is_claude:
        click.echo(f"  backend: claude ({CONTEXTUAL_CLAUDE_MODEL})")
        click.echo(f"  pricing: ${HAIKU_INPUT_PER_M}/1M input · ${HAIKU_OUTPUT_PER_M}/1M output")
        click.echo(f"  cache write: {HAIKU_CACHE_WRITE_MULT}x · cache read: {HAIKU_CACHE_READ_MULT}x")
    else:
        model = cfg.llm.ollama_model
        measured = "measured" if estimate.tok_per_sec_measured else "estimated — no benchmark"
        click.echo(f"  backend: ollama ({model}) · no API cost")
        click.echo(f"  throughput: {estimate.tok_per_sec:.0f} tok/s ({measured})")

    click.echo(f"  contextual_max_doc_tokens: {cfg.ingestion.contextual_max_doc_tokens}")
    click.echo(f"  contextual_window_chunks: {cfg.ingestion.contextual_window_chunks}")
    if estimate.already_done:
        click.echo(f"  already contextualized: {estimate.already_done} files (skipped)")
    click.echo()


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"~{s:.0f}s"
    m = int(s) // 60
    sec = int(s) % 60
    return f"~{m}m {sec:02d}s"


# ---------------------------------------------------------------------------
# Contextual generation
# ---------------------------------------------------------------------------

def contextualize_files(
    store: VectorStore,
    encoder: Encoder,
    meta_db: MetaDB,
    cfg: Config,
) -> None:
    """Run contextual generation on all un-contextualized files."""
    uncontext = meta_db.get_uncontext_files()
    if not uncontext:
        click.echo("All files already contextualized.")
        return

    backend = cfg.ingestion.contextual_backend
    max_doc_tokens = cfg.ingestion.contextual_max_doc_tokens
    window_chunks = cfg.ingestion.contextual_window_chunks
    docs_dir = cfg.paths.documents_dir

    click.echo(f"Contextualizing {len(uncontext)} file(s) via {backend} ...")

    for i, file_row in enumerate(uncontext, 1):
        source_key = file_row["source_key"]
        file_path = docs_dir / source_key

        if not file_path.exists():
            logger.warning("Source file missing: %s — skipping", source_key)
            meta_db.mark_source_missing(source_key, True)
            continue

        click.echo(f"  [{i}/{len(uncontext)}] {source_key} ...")

        try:
            _contextualize_one_file(
                source_key, file_path, store, encoder, meta_db, cfg,
                backend=backend,
                max_doc_tokens=max_doc_tokens,
                window_chunks=window_chunks,
            )
        except Exception:
            logger.exception("Failed to contextualize %s — continuing", source_key)
            click.echo(f"    ERROR: failed (see logs). Continuing to next file.")


def _contextualize_one_file(
    source_key: str,
    file_path: Path,
    store: VectorStore,
    encoder: Encoder,
    meta_db: MetaDB,
    cfg: Config,
    *,
    backend: str,
    max_doc_tokens: int,
    window_chunks: int,
) -> None:
    """Contextualize all chunks of a single file."""
    from .registry import extract_file

    # Extract full document text
    pages = extract_file(file_path, cfg.ingestion)
    if not pages:
        logger.warning("No text extracted from %s", source_key)
        return

    full_text = "\n\n".join(p.text for p in pages if p.text)
    token_count = len(full_text) // 4
    use_sliding = token_count > max_doc_tokens

    # Fetch all chunks for this file from ChromaDB
    all_chunks = _get_file_chunks(store, source_key)
    if not all_chunks:
        logger.warning("No chunks in store for %s", source_key)
        return

    total_tokens_used = 0
    total_cost = 0.0
    total_tok_per_sec_samples: list[float] = []

    for idx, chunk in enumerate(all_chunks):
        chunk_id = chunk["id"]
        chunk_text = chunk["text"]

        if use_sliding:
            # Build window from surrounding chunks
            start = max(0, idx - window_chunks)
            end = min(len(all_chunks), idx + window_chunks + 1)
            window_texts = [all_chunks[j]["text"] for j in range(start, end) if j != idx]
            context_input = "\n\n---\n\n".join(window_texts)
            prompt = _CONTEXT_PROMPT_WINDOW.format(
                window_text=context_input, chunk_text=chunk_text,
            )
        else:
            prompt = _CONTEXT_PROMPT.format(
                doc_text=full_text, chunk_text=chunk_text,
            )

        # Call LLM
        if backend == "claude":
            context_str, tokens_used, cost = _claude_context_call(
                prompt, full_text if not use_sliding else None, cfg,
            )
            total_tokens_used += tokens_used
            total_cost += cost
        elif backend == "ollama":
            context_str, tokens_used, tok_per_sec = _ollama_context_call(prompt, cfg)
            total_tokens_used += tokens_used
            if tok_per_sec > 0:
                total_tok_per_sec_samples.append(tok_per_sec)
        else:
            raise ValueError(f"Unknown contextual backend: {backend}")

        # Prepend context to chunk text and re-embed
        new_text = context_str.strip() + "\n\n" + chunk_text
        new_embedding = encoder.encode([new_text])[0]

        # Update in ChromaDB (same ID, new text + vector)
        store._collection.update(
            ids=[chunk_id],
            embeddings=[new_embedding],
            documents=[new_text],
        )

    # Record in SQLite
    meta_db.mark_contextual_done(
        source_key=source_key,
        backend=backend,
        tokens_used=total_tokens_used,
        cost_usd=total_cost,
        used_sliding_window=use_sliding,
    )

    # Ollama benchmark
    if backend == "ollama" and total_tok_per_sec_samples:
        avg_tps = sum(total_tok_per_sec_samples) / len(total_tok_per_sec_samples)
        meta_db.upsert_benchmark(
            model=cfg.llm.ollama_model,
            tok_per_sec=avg_tps,
            sample_chunks=len(total_tok_per_sec_samples),
        )

    cost_str = f" (${total_cost:.4f})" if backend == "claude" else ""
    click.echo(f"    → {len(all_chunks)} chunks contextualized{cost_str}")


def _get_file_chunks(store: VectorStore, source_key: str) -> list[dict]:
    """Get all chunks for a file, ordered by chunk_index."""
    results = store._collection.get(
        where={K_SOURCE: source_key},
        include=["documents", "metadatas"],
    )
    chunks = []
    for i, cid in enumerate(results["ids"]):
        chunks.append({
            "id": cid,
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    # Sort by chunk_index for sliding window consistency
    chunks.sort(key=lambda c: int(c["metadata"].get("chunk_index", 0)))
    return chunks


def _claude_context_call(
    prompt: str,
    full_doc_text: str | None,
    cfg: Config,
) -> tuple[str, int, float]:
    """Call Claude for one context generation.  Returns (context, tokens_used, cost_usd)."""
    from anthropic import Anthropic  # type: ignore[import-untyped]

    client = Anthropic(api_key=cfg.llm.claude_api_key)

    messages_content: list[dict] = []
    if full_doc_text is not None:
        # Use cache_control on the document text for prompt caching
        messages_content.append({
            "type": "text",
            "text": full_doc_text,
            "cache_control": {"type": "ephemeral"},
        })
        # The actual chunk prompt follows
        # Strip the document part from prompt since we sent it separately
        chunk_prompt = prompt.split("</document>")[-1].strip()
        messages_content.append({"type": "text", "text": chunk_prompt})
    else:
        messages_content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model=CONTEXTUAL_CLAUDE_MODEL,
        max_tokens=150,
        temperature=0.0,
        messages=[{"role": "user", "content": messages_content}],
    )

    text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text += getattr(block, "text", "")

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

    # Calculate cost
    regular_input = input_tokens - cache_read - cache_creation
    cost = (
        regular_input * HAIKU_INPUT_PER_M / 1_000_000
        + cache_creation * HAIKU_INPUT_PER_M * HAIKU_CACHE_WRITE_MULT / 1_000_000
        + cache_read * HAIKU_INPUT_PER_M * HAIKU_CACHE_READ_MULT / 1_000_000
        + output_tokens * HAIKU_OUTPUT_PER_M / 1_000_000
    )

    return text.strip(), input_tokens + output_tokens, cost


def _ollama_context_call(
    prompt: str,
    cfg: Config,
) -> tuple[str, int, float]:
    """Call Ollama for one context generation.  Returns (context, tokens_used, tok_per_sec)."""
    import ollama  # type: ignore[import-untyped]

    client = ollama.Client(host=cfg.llm.ollama_host)
    start = time.perf_counter()
    response = client.chat(
        model=cfg.llm.ollama_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 150},
    )
    elapsed = time.perf_counter() - start

    text = response["message"]["content"].strip()
    eval_count = response.get("eval_count", 0) or 0
    tok_per_sec = eval_count / elapsed if elapsed > 0 else 0.0

    return text, eval_count, tok_per_sec
