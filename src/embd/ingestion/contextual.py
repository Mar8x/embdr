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
MLX_DEFAULT_TOK_S = 60.0  # MLX on Apple Silicon is typically faster than Ollama

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
    ctx_cfg = cfg.ingestion.contextual
    backend = ctx_cfg.backend
    max_doc_tokens = ctx_cfg.max_doc_tokens
    window_chunks = ctx_cfg.window_chunks

    all_files = meta_db.get_all_files()
    uncontext = [f for f in all_files.values() if not f["contextual_done"] and not f["source_missing"]]
    already_done = sum(1 for f in all_files.values() if f["contextual_done"])

    default_tps = MLX_DEFAULT_TOK_S if backend == "mlx" else OLLAMA_DEFAULT_TOK_S
    result = EstimateResult(backend=backend, already_done=already_done, tok_per_sec=default_tps)

    # Local backend throughput lookup
    if backend in ("ollama", "mlx"):
        bench_model = ctx_cfg.ollama_model if backend == "ollama" else ctx_cfg.mlx_model
        bench = meta_db.get_benchmark(bench_model)
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
        elif backend in ("ollama", "mlx"):
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
    name_w = 40  # max display width for filename column

    click.echo(f"\n  {'file':<{name_w}s} {'chunks':>6s}   {'chars':>10s}   {'tokens':>8s}   {'window':<8s} {header_last:>10s}")
    click.echo("  " + "─" * (name_w + 50))

    total_cost = 0.0
    total_time = 0.0
    total_chunks = 0
    total_chars = 0
    total_tokens = 0

    for f in estimate.files:
        name = _truncate(f.source_key, name_w)
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
            f"  {name:<{name_w}s} {f.chunk_count:>6d}   {f.char_count:>10,d}   "
            f"{f.token_count:>8,d}   {window_str:<8s} {val_str:>10s}"
        )

    click.echo("  " + "─" * (name_w + 50))
    total_val = f"${total_cost:.2f}" if is_claude else _fmt_seconds(total_time)
    click.echo(
        f"  {'total':<{name_w}s} {total_chunks:>6d}   {total_chars:>10,d}   "
        f"{total_tokens:>8,d}   {'':<8s} {total_val:>10s}"
    )
    click.echo("  " + "─" * (name_w + 50))
    click.echo(f"  {'file':<{name_w}s} {'chunks':>6s}   {'chars':>10s}   {'tokens':>8s}   {'window':<8s} {header_last:>10s}")

    click.echo()
    ctx_cfg = cfg.ingestion.contextual
    if is_claude:
        click.echo(f"  backend: claude ({ctx_cfg.claude_model})")
        click.echo(f"  pricing: ${HAIKU_INPUT_PER_M}/1M input · ${HAIKU_OUTPUT_PER_M}/1M output")
        click.echo(f"  cache write: {HAIKU_CACHE_WRITE_MULT}x · cache read: {HAIKU_CACHE_READ_MULT}x")
    elif estimate.backend == "ollama":
        measured = "measured" if estimate.tok_per_sec_measured else f"default {OLLAMA_DEFAULT_TOK_S:.0f} — run once to calibrate"
        click.echo(f"  backend: ollama ({ctx_cfg.ollama_model}) · no API cost")
        click.echo(f"  throughput: {estimate.tok_per_sec:.0f} tok/s ({measured})")
        click.echo(f"  formula: chunks × {CONTEXT_OUTPUT_TOKENS} output tokens / throughput")
    else:  # mlx
        measured = "measured" if estimate.tok_per_sec_measured else f"default {MLX_DEFAULT_TOK_S:.0f} — run once to calibrate"
        click.echo(f"  backend: mlx ({ctx_cfg.mlx_model}) · no API cost")
        click.echo(f"  throughput: {estimate.tok_per_sec:.0f} tok/s ({measured})")
        click.echo(f"  formula: chunks × {CONTEXT_OUTPUT_TOKENS} output tokens / throughput")

    click.echo(f"  max_doc_tokens: {ctx_cfg.max_doc_tokens}")
    click.echo(f"  window_chunks: {ctx_cfg.window_chunks}")
    if estimate.already_done:
        click.echo(f"  already contextualized: {estimate.already_done} files (skipped)")
    click.echo()


def _truncate(s: str, width: int) -> str:
    """Shorten a string to *width*, preserving the file extension."""
    if len(s) <= width:
        return s
    # Keep the extension visible (e.g. ".pdf")
    dot = s.rfind(".")
    if dot > 0:
        ext = s[dot:]                       # ".pdf"
        stem_budget = width - len(ext) - 1  # room for "…"
        if stem_budget > 4:
            return s[:stem_budget] + "…" + ext
    return s[: width - 1] + "…"


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
    """Contextualize all un-contextualized files (pass 2 of ingest).

    Called after all files have been embedded (pass 1).  For each file that
    lacks context, re-parses the source from disk (~0.1s, negligible), sends
    each chunk + full document text to an LLM, prepends the returned context,
    re-embeds, and updates ChromaDB in place.

    Running this as a separate pass ensures only one model (the contextual
    LLM) uses the GPU at a time — no memory contention with the embedding
    model from pass 1.

    Progress is tracked per-file in SQLite (``contextual_done``).
    Interrupted runs resume from the first incomplete file.
    """
    uncontext = meta_db.get_uncontext_files()
    if not uncontext:
        click.echo("All files already contextualized.")
        return

    ctx_cfg = cfg.ingestion.contextual
    backend = ctx_cfg.backend
    max_doc_tokens = ctx_cfg.max_doc_tokens
    window_chunks = ctx_cfg.window_chunks
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
    import sys

    from .registry import extract_file

    # Extract full document text (~0.1s, negligible vs LLM calls)
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

    n_chunks = len(all_chunks)
    strategy = "sliding window" if use_sliding else "full doc"
    click.echo(f"    {n_chunks} chunks, {strategy}")

    # --- Phase A: generate context strings (LLM only, no embedding model) ---
    total_tokens_used = 0
    total_cost = 0.0
    total_tok_per_sec_samples: list[float] = []
    new_texts: list[str] = []
    ctx_start = time.perf_counter()

    # MLX full-doc mode: share a prompt cache across all chunks so the
    # document prefix KV state is computed once and reused per chunk.
    # Sliding-window mode skips this (each chunk has a unique window prefix).
    mlx_doc_cache: list | None = [] if (backend == "mlx" and not use_sliding) else None

    for idx, chunk in enumerate(all_chunks):
        sys.stderr.write(f"\r    generating {idx + 1}/{n_chunks}")
        sys.stderr.flush()

        chunk_text = chunk["text"]

        if use_sliding:
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
        elif backend == "mlx":
            context_str, tokens_used, tok_per_sec = _mlx_context_call(prompt, cfg, mlx_doc_cache)
            total_tokens_used += tokens_used
            if tok_per_sec > 0:
                total_tok_per_sec_samples.append(tok_per_sec)
        else:
            raise click.ClickException(
                f"Unknown contextual backend: '{backend}'. "
                f"Use 'claude', 'ollama', or 'mlx' in [ingestion.contextual] backend."
            )

        new_texts.append(context_str.strip() + "\n\n" + chunk_text)

    gen_elapsed = time.perf_counter() - ctx_start
    sys.stderr.write("\r" + " " * 40 + "\r")
    sys.stderr.flush()

    # --- Release LLM before re-embedding ---
    if backend == "mlx" and _mlx_cache:
        _mlx_cache.clear()
        import gc; gc.collect()

    # --- Phase B: re-embed all contextualized texts and update ChromaDB ---
    # The LLM is no longer active; embedding model gets full GPU.
    sys.stderr.write(f"\r    re-embedding {n_chunks} chunks ...")
    sys.stderr.flush()
    embed_start = time.perf_counter()
    new_embeddings = encoder.encode(new_texts)
    embed_elapsed = time.perf_counter() - embed_start
    sys.stderr.write("\r" + " " * 40 + "\r")
    sys.stderr.flush()

    chunk_ids = [c["id"] for c in all_chunks]
    store._collection.update(
        ids=chunk_ids,
        embeddings=new_embeddings,
        documents=new_texts,
    )

    # Record in SQLite
    meta_db.mark_contextual_done(
        source_key=source_key,
        backend=backend,
        tokens_used=total_tokens_used,
        cost_usd=total_cost,
        used_sliding_window=use_sliding,
    )

    # Local backend benchmark
    ctx_cfg = cfg.ingestion.contextual
    if backend in ("ollama", "mlx") and total_tok_per_sec_samples:
        bench_model = ctx_cfg.ollama_model if backend == "ollama" else ctx_cfg.mlx_model
        avg_tps = sum(total_tok_per_sec_samples) / len(total_tok_per_sec_samples)
        meta_db.upsert_benchmark(
            model=bench_model,
            tok_per_sec=avg_tps,
            sample_chunks=len(total_tok_per_sec_samples),
        )

    # Summary line
    cost_str = f", ${total_cost:.4f}" if backend == "claude" else ""
    avg_str = ""
    if total_tok_per_sec_samples:
        avg_tps = sum(total_tok_per_sec_samples) / len(total_tok_per_sec_samples)
        avg_str = f", {avg_tps:.0f} tok/s"
    click.echo(
        f"    ✓ {n_chunks} chunks — generate {_fmt_seconds(gen_elapsed)}{avg_str}, "
        f"embed {embed_elapsed:.1f}s{cost_str}"
    )


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
    from anthropic import Anthropic, AuthenticationError, APIError  # type: ignore[import-untyped]

    api_key = cfg.llm.claude_api_key
    if not api_key:
        raise click.ClickException(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to .env or switch [ingestion.contextual] backend to 'ollama'."
        )

    client = Anthropic(api_key=api_key)
    model = cfg.ingestion.contextual.claude_model

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

    try:
        response = client.messages.create(
            model=model,
            max_tokens=150,
            temperature=0.0,
            messages=[{"role": "user", "content": messages_content}],
        )
    except AuthenticationError:
        raise click.ClickException(
            "ANTHROPIC_API_KEY is invalid. Check your .env file."
        ) from None
    except APIError as exc:
        raise click.ClickException(
            f"Claude API error: {exc}"
        ) from None

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

    model = cfg.ingestion.contextual.ollama_model
    client = ollama.Client(host=cfg.llm.ollama_host)  # shared Ollama server
    start = time.perf_counter()
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 150},
            keep_alive=-1,  # keep model + KV cache hot between sequential chunk calls
        )
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise click.ClickException(
                f"Ollama model '{model}' not found. "
                f"Pull it first:  ollama pull {model}\n"
                f"Or change [ingestion.contextual] ollama_model in config.toml."
            ) from None
        raise click.ClickException(
            f"Ollama error ({exc.status_code}): {exc}"
        ) from None
    except Exception as exc:
        host = cfg.llm.ollama_host
        raise click.ClickException(
            f"Cannot reach Ollama at {host}: {exc}\n"
            f"Is 'ollama serve' running?"
        ) from None
    elapsed = time.perf_counter() - start

    text = response["message"]["content"].strip()
    eval_count = response.get("eval_count", 0) or 0
    # Use Ollama's own eval_duration (generation only, nanoseconds) so prefill
    # time from large input contexts doesn't deflate the tok/s measurement.
    eval_duration_ns = response.get("eval_duration", 0) or 0
    if eval_duration_ns > 0:
        tok_per_sec = eval_count / (eval_duration_ns / 1e9)
    elif elapsed > 0:
        tok_per_sec = eval_count / elapsed  # fallback
    else:
        tok_per_sec = 0.0

    return text, eval_count, tok_per_sec


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------

# Module-level cache: model + tokenizer are loaded once and reused across
# all chunks in a run.  Key = model path string.
_mlx_cache: dict[str, tuple] = {}


def _mlx_context_call(
    prompt: str,
    cfg: Config,
    doc_cache: list | None = None,
) -> tuple[str, int, float]:
    """Call MLX for one context generation.  Returns (context, tokens_used, tok_per_sec).

    Requires Apple Silicon and ``pip install 'embd[mlx]'``.

    ``doc_cache`` is a mutable list used as a container for a per-document
    prompt cache object.  Pass an empty list ``[]`` for the first chunk of a
    document; it is populated on first call and reused for subsequent chunks,
    allowing mlx-lm to skip re-prefilling the shared document prefix each time.
    Pass ``None`` to disable caching (e.g. sliding-window mode).
    """
    model_path = cfg.ingestion.contextual.mlx_model

    # Lazy-load once per model
    if model_path not in _mlx_cache:
        try:
            from mlx_lm import load  # type: ignore[import-untyped]
        except ImportError:
            raise click.ClickException(
                "MLX backend requires mlx-lm.  Install with:  pip install 'embd[mlx]'"
            ) from None
        logger.info("Loading MLX model '%s' for contextual generation …", model_path)
        _mlx_cache[model_path] = load(model_path)

    model, tokenizer = _mlx_cache[model_path]

    from mlx_lm import generate as mlx_generate  # type: ignore[import-untyped]
    from mlx_lm.sample_utils import make_sampler  # type: ignore[import-untyped]

    # Lazy-create the prompt cache on the first chunk of each document
    if doc_cache is not None and not doc_cache:
        try:
            from mlx_lm import make_prompt_cache  # type: ignore[import-untyped]
            doc_cache.append(make_prompt_cache(model))
        except Exception:
            doc_cache.append(None)  # mlx-lm version doesn't support it; fall back

    cache = doc_cache[0] if doc_cache else None

    # Format as chat if the tokenizer supports it
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        formatted = prompt

    start = time.perf_counter()
    output: str = mlx_generate(
        model, tokenizer, prompt=formatted,
        max_tokens=150,
        sampler=make_sampler(temp=0.0),
        verbose=False,
        prompt_cache=cache,
    )
    elapsed = time.perf_counter() - start

    try:
        n_out = len(tokenizer.encode(output))
    except Exception:
        n_out = len(output) // 4
    tok_per_sec = n_out / elapsed if elapsed > 0 else 0.0

    return output.strip(), n_out, tok_per_sec
