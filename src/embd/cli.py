"""
Command-line interface for embd.

Commands:
  ingest      — scan documents folder, ingest new/changed PDFs
  ingest-url  — fetch a web page and ingest it into the vector store
  query       — ask a question against indexed documents
  delete      — remove one file's chunks from the vector store
  rebuild     — drop everything and re-ingest from scratch
  shell       — interactive Textual TUI for Q&A
  serve       — HTTP retrieval API (ChatGPT Actions / OpenAI-style /query)

Each command prints a performance report at the end showing timing
and memory usage, so you can see exactly where time is spent.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from .config import Config, load_config
from .display_format import format_local_sources_footer
from .embedding.encoder import Encoder
from .ingestion.chunker import chunk_pages
from .ingestion.extractor import extract_pages
from .ingestion.ingest import ingest_file as _ingest_file_core
from .ingestion.registry import get_supported_extensions
from .ingestion.scanner import hash_file, scan_documents
from .ingestion.epub_extractor import extract_epub_pages
from .ingestion.web_extractor import extract_url, fetch_and_extract, url_to_source_name
from .perf import Timer, print_ingest_report, print_query_report
from .qa.retriever import RetrievedChunk
from .store.vector_store import (
    K_CHUNK_IDX, K_CHUNK_OVL, K_CHUNK_SIZE,
    K_DOC_DATE, K_DOC_DATE_Q,
    K_HASH, K_INGESTED, K_MODEL, K_PAGE, K_SOURCE,
    VectorStore,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Silence noisy third-party loggers unless in verbose mode
    if not verbose:
        for noisy in ("sentence_transformers", "chromadb", "httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Root command group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--config", "config_path",
    default="config.toml",
    show_default=True,
    type=click.Path(exists=True),
    help="Path to config.toml",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, config_path: str, verbose: bool) -> None:
    """embd — fully local document Q&A, optimized for Apple Silicon."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(Path(config_path))


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--refresh", is_flag=True,
              help="Re-check all files by hash; skip unchanged, re-ingest changed.")
@click.option("--reset", is_flag=True,
              help="Drop all embeddings and re-ingest from scratch (implies --refresh).")
@click.option("--contextualize", is_flag=True,
              help="After embedding, contextualize all un-contextualized files "
                   "(new + previously ingested). Already-done files are skipped.")
@click.option("--recontext", is_flag=True,
              help="Re-contextualize ALL files from scratch (resets contextual state first).")
@click.option("--recontext-file", type=str, default=None, metavar="SOURCE_KEY",
              help="Re-contextualize a single file by source key (e.g. 'report.pdf').")
@click.option("--contextualize-estimate-only", is_flag=True,
              help="Print cost/time estimate for contextual generation, then exit.")
@click.option("-y", "--yes", is_flag=True,
              help="Skip confirmation prompt and proceed immediately.")
@click.pass_context
def ingest(
    ctx: click.Context,
    refresh: bool,
    reset: bool,
    contextualize: bool,
    recontext: bool,
    recontext_file: str | None,
    contextualize_estimate_only: bool,
    yes: bool,
) -> None:
    """Scan the documents folder and ingest new or changed files.

    \b
    embd ingest                              embed new/changed files
    embd ingest --contextualize              embed, then contextualize un-done files
    embd ingest --recontext                  re-contextualize ALL files from scratch
    embd ingest --recontext-file report.pdf  re-contextualize one specific file
    embd ingest -y                           skip confirmation prompt

    --contextualize processes every file not yet contextualized (new +
    previously ingested). --recontext resets all contextual state first,
    then contextualizes everything. Interrupted runs resume where they left off.
    """
    cfg: Config = ctx.obj["config"]
    from .ingestion.ingest import _source_key
    from .ingestion.bm25_index import build_and_save_bm25
    from .store.meta_db import MetaDB

    # --reset implies --refresh
    if reset:
        refresh = True

    # Ensure documents dir exists
    cfg.paths.documents_dir.mkdir(parents=True, exist_ok=True)

    store   = VectorStore(cfg.paths.db_dir, cfg.retrieval.collection_name)
    meta_db = MetaDB(cfg.paths.db_dir / "embd_meta.db")
    docs_dir = cfg.paths.documents_dir

    # --recontext / --recontext-file imply --contextualize
    if recontext or recontext_file:
        contextualize = True

    # --recontext-file: re-contextualize a single file and exit
    if recontext_file:
        row = meta_db.get_file(recontext_file)
        if not row:
            click.echo(f"File '{recontext_file}' not found in index. Run `embd ingest` first.")
            meta_db.close()
            return
        file_path = cfg.paths.documents_dir / recontext_file
        if not file_path.exists():
            click.echo(f"Source file not found on disk: {file_path}")
            meta_db.close()
            return
        meta_db.reset_contextual(recontext_file)
        click.echo(f"Reset contextual state for '{recontext_file}'.")
        encoder = Encoder(cfg.embedding.model_name, cfg.embedding.device)
        from .ingestion.contextual import _contextualize_one_file
        ctx_cfg = cfg.ingestion.contextual
        _contextualize_one_file(
            recontext_file, file_path, store, encoder, meta_db, cfg,
            backend=ctx_cfg.backend,
            max_doc_tokens=ctx_cfg.max_doc_tokens,
            window_chunks=ctx_cfg.window_chunks,
        )
        meta_db.close()
        return

    # --recontext: reset ALL contextual state
    if recontext:
        n_reset = sum(1 for f in meta_db.get_all_files().values() if f.get("contextual_done"))
        if n_reset:
            if not yes and sys.stdin.isatty():
                click.confirm(
                    f"This will reset contextual state for {n_reset} files and re-contextualize all.\n"
                    "Continue?",
                    abort=True,
                )
            meta_db.reset_all_contextual()
            click.echo(f"Reset contextual state for {n_reset} files.")

    # --contextualize-estimate-only: print estimate and exit
    if contextualize_estimate_only:
        from .ingestion.contextual import estimate_contextualization, print_estimate
        estimate = estimate_contextualization(meta_db, cfg)
        print_estimate(estimate, cfg)
        meta_db.close()
        return

    # --reset: drop everything, prompt for confirmation
    if reset:
        if not yes and sys.stdin.isatty():
            click.confirm(
                "This will DELETE all embeddings and re-ingest from scratch.\n"
                f"  DB: {cfg.paths.db_dir}\n"
                "Continue?",
                abort=True,
            )
        store._client.delete_collection(cfg.retrieval.collection_name)
        store._collection = store._client.get_or_create_collection(
            name=cfg.retrieval.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        meta_db.reset()
        click.echo(f"Cleared collection '{cfg.retrieval.collection_name}'.")

    # ---- Scan ----
    # For --refresh/--reset, ignore stored state to force re-check
    if refresh:
        known_files: dict[str, str] = {}
    else:
        known_files = store.get_known_files()

    exts = get_supported_extensions(cfg.ingestion.enabled_types)
    scan = scan_documents(
        cfg.paths.documents_dir, known_files, exts,
        ignore_patterns=cfg.ingestion.ignore_patterns,
    )

    to_process = scan.to_add + scan.to_update
    n_existing = len(known_files) - len(scan.to_update) - len(scan.to_remove)

    # ---- Nothing new to embed ----
    if not to_process and not scan.to_remove:
        if not contextualize:
            click.echo("Nothing to ingest — all files are up to date.")
            meta_db.close()
            return
        # --contextualize with nothing new: run pass 2 only
        uncontext = meta_db.get_uncontext_files()
        if not uncontext:
            click.echo("Nothing to ingest and all files already contextualized.")
            meta_db.close()
            return
        click.echo(f"Nothing new to embed. {len(uncontext)} file(s) to contextualize.")
        if not yes and sys.stdin.isatty():
            click.confirm("Proceed?", abort=True)
        encoder = Encoder(cfg.embedding.model_name, cfg.embedding.device)
        from .ingestion.contextual import contextualize_files
        contextualize_files(store, encoder, meta_db, cfg)
        meta_db.close()
        return

    # Count files by extension for the summary
    from collections import Counter
    ext_new: Counter[str] = Counter(p.suffix.lower() for p in scan.to_add)
    ext_upd: Counter[str] = Counter(p.suffix.lower() for p in scan.to_update)

    click.echo()
    click.echo("  ── scan ──")
    click.echo(f"  documents dir:  {cfg.paths.documents_dir}")
    click.echo(f"  unchanged:      {n_existing} files (skipped)")
    if scan.to_add:
        types = ", ".join(f"{e} {n}" for e, n in sorted(ext_new.items()))
        click.echo(f"  new:            {len(scan.to_add)} files  ({types})")
    if scan.to_update:
        types = ", ".join(f"{e} {n}" for e, n in sorted(ext_upd.items()))
        click.echo(f"  changed:        {len(scan.to_update)} files  ({types})")
    if scan.to_remove:
        click.echo(f"  removed:        {len(scan.to_remove)} files")

    click.echo()
    click.echo("  ── embed ──")
    click.echo(f"  model:          {cfg.embedding.model_name}")
    click.echo(f"  device:         {cfg.embedding.device}")
    click.echo(f"  chunk size:     {cfg.ingestion.chunk_size} chars, {cfg.ingestion.chunk_overlap} overlap")
    ocr_mode = cfg.ingestion.ocr_embedded_images
    click.echo(f"  image OCR:      {ocr_mode} (backend={cfg.ingestion.ocr_backend})")

    if contextualize:
        ctx_cfg = cfg.ingestion.contextual
        model_name = {
            "ollama": ctx_cfg.ollama_model,
            "claude": ctx_cfg.claude_model,
            "mlx": ctx_cfg.mlx_model,
        }.get(ctx_cfg.backend, ctx_cfg.backend)
        # Count how many files will need contextualizing after pass 1
        n_already_ctx = sum(
            1 for f in meta_db.get_all_files().values() if f.get("contextual_done")
        )
        n_will_ctx = (n_existing - n_already_ctx) + len(to_process)
        click.echo()
        click.echo("  ── contextualize ──")
        click.echo(f"  backend:        {ctx_cfg.backend}")
        click.echo(f"  model:          {model_name}")
        click.echo(f"  to process:     {n_will_ctx} files  ({n_already_ctx} already done, skipped)")
        strategy = f"sliding window (>{ctx_cfg.max_doc_tokens} tokens)"
        click.echo(f"  large docs:     {strategy}")
    click.echo()

    # ---- Confirm ----
    if not yes and sys.stdin.isatty():
        click.confirm("Proceed?", abort=True)

    # ---- Pass 1: embed ----
    encoder = Encoder(cfg.embedding.model_name, cfg.embedding.device)

    with Timer("total") as total_timer:
        # Delete stale data before re-ingesting changed files
        for path in scan.to_update:
            sk = _source_key(path, docs_dir)
            store.delete_file(sk)
            meta_db.reset_contextual(sk)

        # Remove files that no longer exist on disk
        for name in scan.to_remove:
            store.delete_file(name)
            meta_db.remove_file(name)

        total_chunks = 0
        embed_elapsed = 0.0

        for i, pdf_path in enumerate(to_process, start=1):
            src_key = _source_key(pdf_path, docs_dir)
            click.echo(f"[{i}/{len(to_process)}] {src_key} ...")
            result = _ingest_file_core(
                pdf_path, store, encoder, cfg, source_name=src_key,
            )
            if result is None:
                click.echo("  skipped (no extractable text)")
                continue
            total_chunks += result.n_chunks
            embed_elapsed += result.embed_s
            click.echo(f"  → {result.n_chunks} chunks, embed {result.embed_s:.1f}s")

            # Sync to SQLite
            mtime = pdf_path.stat().st_mtime
            file_hash = hash_file(pdf_path)
            meta_db.upsert_file(
                source_key=src_key,
                file_hash=file_hash,
                mtime=mtime,
                char_count=result.char_count,
                chunk_count=result.n_chunks,
                file_type=pdf_path.suffix.lower(),
                extract_s=result.extract_s,
                embed_s=result.embed_s,
                upsert_s=result.upsert_s,
                embedding_model=cfg.embedding.model_name,
            )

        print_ingest_report(
            n_new=len(scan.to_add),
            n_updated=len(scan.to_update),
            n_removed=len(scan.to_remove),
            n_chunks=total_chunks,
            embed_s=embed_elapsed,
            total_s=total_timer.elapsed,
        )

        # Build BM25 index after any modifications
        click.echo("Building BM25 index ...")
        with Timer("bm25") as bm25_timer:
            build_and_save_bm25(store, cfg.paths.db_dir)
        click.echo(f"BM25 index built in {bm25_timer.elapsed:.1f}s")

    # ---- Pass 2: contextualize (only when explicitly requested) ----
    # Each file is processed in two phases: LLM generation (phase A), then
    # re-embedding (phase B).  Only one model uses the GPU at a time.
    if contextualize:
        from .ingestion.contextual import contextualize_files
        contextualize_files(store, encoder, meta_db, cfg)

    meta_db.close()


# ---------------------------------------------------------------------------
# ingest-url
# ---------------------------------------------------------------------------

@cli.command("ingest-url")
@click.argument("url")
@click.pass_context
def ingest_url(ctx: click.Context, url: str) -> None:
    """Fetch a web page and ingest its text into the vector store.

    Works with any public URL — .edu sites, blog posts, documentation, etc.
    The page content is chunked and embedded just like PDF pages.
    """
    import hashlib

    cfg: Config = ctx.obj["config"]
    store = VectorStore(cfg.paths.db_dir, cfg.retrieval.collection_name)
    encoder = Encoder(cfg.embedding.model_name, cfg.embedding.device)

    result = fetch_and_extract(url)
    click.echo(result.summary())

    if not result.ok:
        return

    source_name = url_to_source_name(url)
    store.delete_file(source_name)

    chunks = chunk_pages(
        result.pages,
        source_filename=source_name,
        chunk_size=cfg.ingestion.chunk_size,
        chunk_overlap=cfg.ingestion.chunk_overlap,
    )
    texts = [c.text for c in chunks]
    click.echo(f"  → {len(chunks)} chunks to embed")

    with Timer("embed") as embed_timer:
        embeddings = encoder.encode(texts)

    now = datetime.now(timezone.utc).isoformat()
    content_hash = hashlib.sha256(texts[0].encode()).hexdigest()
    doc_date = result.document_date
    metadatas = [
        {
            K_SOURCE:     source_name,
            K_PAGE:       c.page_num,
            K_CHUNK_IDX:  c.chunk_index,
            K_INGESTED:   now,
            K_HASH:       content_hash,
            K_MODEL:      encoder.model_version,
            K_CHUNK_SIZE: cfg.ingestion.chunk_size,
            K_CHUNK_OVL:  cfg.ingestion.chunk_overlap,
            K_DOC_DATE:   (doc_date.date or "") if doc_date else "",
            K_DOC_DATE_Q: (doc_date.quality) if doc_date else "none",
        }
        for c in chunks
    ]

    store.upsert_chunks(
        chunk_ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        texts=texts,
        metadatas=metadatas,
    )
    click.echo(
        f"Ingested {len(chunks)} chunks in {embed_timer.elapsed:.1f}s "
        f"(source: {source_name})"
    )


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("question")
@click.option("--diag", is_flag=True, help="Show retrieval diagnostics (RRF scores, sources).")
@click.pass_context
def query(ctx: click.Context, question: str, diag: bool) -> None:
    """Ask a question against all indexed documents.

    Retrieves the most relevant chunks, then generates a grounded answer
    using only that content. Sources are listed below the answer.
    """
    cfg: Config = ctx.obj["config"]

    store   = VectorStore(cfg.paths.db_dir, cfg.retrieval.collection_name)
    encoder = Encoder(cfg.embedding.model_name, cfg.embedding.device)

    # Load BM25 index for hybrid search (graceful: None if absent)
    from .ingestion.bm25_index import BM25Index, BM25_FILENAME
    bm25 = BM25Index.load(cfg.paths.db_dir / BM25_FILENAME)

    # Step 1: Embed the query (timed separately from retrieval)
    with Timer("embed") as embed_timer:
        query_vec = encoder.encode_query(question)
    embed_ms = embed_timer.elapsed * 1000

    # Step 2: Retrieve top-k chunks (hybrid if BM25 available)
    with Timer("retrieve") as retrieve_timer:
        if bm25 is not None and cfg.retrieval.bm25_weight > 0:
            from .qa.hybrid_retriever import rrf_merge
            fetch_k = cfg.retrieval.top_k * 2
            semantic_hits = store.query(query_vec, top_k=fetch_k)
            bm25_hits = bm25.query(question, top_k=fetch_k)
            merge = rrf_merge(
                semantic_hits, bm25_hits, store,
                semantic_weight=cfg.retrieval.semantic_weight,
                bm25_weight=cfg.retrieval.bm25_weight,
                top_k=cfg.retrieval.top_k,
            )
            hits = merge.results
        else:
            merge = None
            hits = store.query(query_vec, top_k=cfg.retrieval.top_k)
    retrieve_ms = retrieve_timer.elapsed * 1000

    if not hits:
        click.echo(
            "No chunks found in the index. "
            "Have you run `embd ingest` yet?",
            err=False,
        )
        return

    chunks = [
        RetrievedChunk(
            text=h["text"],
            source_filename=h["metadata"]["source_filename"],
            page_num=int(h["metadata"]["page_num"]),
            chunk_index=int(h["metadata"]["chunk_index"]),
            distance=h["distance"],
        )
        for h in hits
    ]

    # Step 3: Generate a grounded answer
    generator = _make_generator(cfg)
    with Timer("generate") as gen_timer:
        answer, usage = generator.generate(question, chunks)
    generate_s = gen_timer.elapsed

    footer, _ = format_local_sources_footer(chunks, cfg.paths.documents_dir)

    # Output: answer, then footnote-style sources, then perf report
    click.echo("\n" + "=" * 60)
    click.echo(answer)
    click.echo("=" * 60)
    if footer:
        click.echo(footer.rstrip("\n"))

    print_query_report(
        embed_ms=embed_ms,
        retrieve_ms=retrieve_ms,
        generate_s=generate_s,
        usage=usage,
        top_k=cfg.retrieval.top_k,
    )

    if diag and merge is not None:
        click.echo()
        click.echo("  ── retrieval diagnostics ──")
        click.echo(f"  weights: semantic={cfg.retrieval.semantic_weight}, bm25={cfg.retrieval.bm25_weight}")
        click.echo(f"  {'#':<3s} {'source':<15s} {'sem_rank':>8s} {'bm25_rank':>9s} {'sem':>7s} {'bm25':>7s} {'rrf':>7s}  file")
        for i, d in enumerate(merge.diag, 1):
            sr = str(d.semantic_rank + 1) if d.semantic_rank is not None else "—"
            br = str(d.bm25_rank + 1) if d.bm25_rank is not None else "—"
            # Get source filename from the hit
            src_file = hits[i - 1]["metadata"].get("source_filename", "") if i <= len(hits) else ""
            click.echo(
                f"  {i:<3d} {d.source:<15s} {sr:>8s} {br:>9s} "
                f"{d.semantic_contrib:>7.4f} {d.bm25_contrib:>7.4f} {d.rrf_score:>7.4f}  {src_file}"
            )
    elif diag:
        click.echo("\n  (pure semantic search — no BM25 index loaded, no RRF diagnostics)")


def _make_generator(cfg: Config):
    """Instantiate the appropriate generator based on config.llm.backend."""
    from .qa.generator_mlx import resolve_system_prompt

    backend = cfg.llm.backend.strip().lower()
    system_prompt = resolve_system_prompt(
        cfg.llm.system_prompt_preset,
        cfg.llm.system_prompt,
    )
    if backend == "mlx":
        try:
            from .qa.generator_mlx import MLXGenerator
        except ImportError:
            raise click.ClickException(
                "MLX backend requires mlx-lm. Install with: pip install 'embd[mlx]'"
            )
        return MLXGenerator(
            model_path=cfg.llm.mlx_model,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            system_prompt=system_prompt,
        )
    elif backend == "ollama":
        from .qa.generator_ollama import OllamaGenerator
        return OllamaGenerator(
            model=cfg.llm.ollama_model,
            host=cfg.llm.ollama_host,
            temperature=cfg.llm.temperature,
            num_predict=cfg.llm.max_tokens,
            system_prompt=system_prompt,
        )
    elif backend == "claude":
        from .qa.generator_claude import ClaudeGenerator
        return ClaudeGenerator(
            model=cfg.llm.claude_model,
            api_key=cfg.llm.claude_api_key,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
            system_prompt=system_prompt,
        )
    else:
        raise click.ClickException(
            f"Unknown LLM backend: '{cfg.llm.backend}'. "
            "Set backend to 'mlx', 'ollama', or 'claude' in config.toml."
        )


@cli.command()
@click.pass_context
def shell(ctx: click.Context) -> None:
    """Start the interactive Textual Q&A shell."""
    cfg: Config = ctx.obj["config"]
    from .shell import EmbdShell

    EmbdShell(cfg).run()


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--host",
    default=None,
    help="Bind address (default: [server].host or 0.0.0.0)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Listen port (default: [server].port or 8000)",
)
@click.pass_context
def serve(ctx: click.Context, host: str | None, port: int | None) -> None:
    """Run the OpenAI-compatible retrieval HTTP API (POST /query).

    Requires an API key: set EMBD_API_KEY or ``api_key`` under ``[server]`` in
    config.toml. For HTTPS in production, terminate TLS in nginx (see Docker Compose
    and the README) or another reverse proxy in front of this port.

    ChatGPT custom GPT: point the Action at your public base URL and import
    ``/openapi.json``; authentication type API Key, Bearer.
    """
    cfg: Config = ctx.obj["config"]
    from .server import create_app, resolve_api_key

    if not resolve_api_key(cfg):
        raise click.ClickException(
            "Retrieval API needs an API key. Set environment variable EMBD_API_KEY "
            "or add api_key under [server] in config.toml (e.g. output of "
            "`openssl rand -hex 32`)."
        )

    bind_host = host if host is not None else cfg.server.host
    bind_port = port if port is not None else cfg.server.port

    try:
        app = create_app(cfg)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    import uvicorn

    enabled = ", ".join(cfg.ingestion.enabled_types)
    click.echo(
        f"embd retrieval API on http://{bind_host}:{bind_port}\n"
        f"  POST /query       (Authorization: Bearer …)\n"
        f"  GET  /health      (no auth)\n"
        f"  GET  /docs        Swagger UI (field descriptions + examples)\n"
        f"  GET  /redoc       ReDoc\n"
        f"  GET  /openapi.json\n"
        f"\n"
        f"  Embedding model is warmed at startup (first query is fast).\n"
        f"  File watcher on {cfg.paths.documents_dir}\n"
        f"  Enabled types: {enabled}\n"
        f"  Drop/update/delete files; chunks are upserted automatically.\n"
    )
    uvicorn.run(app, host=bind_host, port=bind_port, log_level="info")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("filename")
@click.pass_context
def delete(ctx: click.Context, filename: str) -> None:
    """Remove all indexed chunks for FILENAME from the vector store.

    FILENAME is the source key shown in retrieval results — either a bare
    filename (e.g. report.pdf) or a relative path (e.g. papers/report.pdf).
    The file itself is not touched.
    """
    cfg: Config = ctx.obj["config"]
    store = VectorStore(cfg.paths.db_dir, cfg.retrieval.collection_name)
    count = store.delete_file(filename)
    if count:
        click.echo(f"Deleted {count} chunk(s) for '{filename}'.")
    else:
        click.echo(f"No indexed chunks found for '{filename}'. Nothing deleted.")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show index statistics: files, chunks, storage, timing, and contextual status."""
    cfg: Config = ctx.obj["config"]
    from .store.meta_db import MetaDB

    store = VectorStore(cfg.paths.db_dir, cfg.retrieval.collection_name)
    meta_db = MetaDB(cfg.paths.db_dir / "embd_meta.db")
    all_files = meta_db.get_all_files()

    total_chunks = store.count()
    n_files = len(all_files)

    if n_files == 0:
        click.echo("No files indexed. Run `embd ingest` first.")
        meta_db.close()
        return

    # Aggregate stats
    total_chars = sum(f.get("char_count") or 0 for f in all_files.values())
    total_tokens = sum(f.get("token_count") or 0 for f in all_files.values())
    n_ctx_done = sum(1 for f in all_files.values() if f.get("contextual_done"))
    n_missing = sum(1 for f in all_files.values() if f.get("source_missing"))

    # By file type
    from collections import Counter, defaultdict
    type_counts: Counter[str] = Counter()
    type_chunks: Counter[str] = Counter()
    type_chars: Counter[str] = Counter()
    for f in all_files.values():
        ft = f.get("file_type") or "unknown"
        type_counts[ft] += 1
        type_chunks[ft] += f.get("chunk_count") or 0
        type_chars[ft] += f.get("char_count") or 0

    # Timing stats (only files with timing data)
    files_with_timing = [f for f in all_files.values() if f.get("embed_s") is not None]
    total_extract = sum(f["extract_s"] for f in files_with_timing if f.get("extract_s"))
    total_embed = sum(f["embed_s"] for f in files_with_timing if f.get("embed_s"))
    total_upsert = sum(f["upsert_s"] for f in files_with_timing if f.get("upsert_s"))

    # Embedding models used
    models_used: Counter[str] = Counter()
    for f in all_files.values():
        m = f.get("embedding_model")
        if m:
            models_used[m] += 1

    # Storage on disk
    db_dir = cfg.paths.db_dir
    db_size_mb = 0.0
    if db_dir.exists():
        for p in db_dir.rglob("*"):
            if p.is_file():
                db_size_mb += p.stat().st_size
        db_size_mb /= 1024 * 1024

    # --- Output ---
    click.echo()
    click.echo("  ── index ──")
    click.echo(f"  files:          {n_files}")
    click.echo(f"  chunks:         {total_chunks:,}")
    click.echo(f"  characters:     {total_chars:,}")
    click.echo(f"  tokens (est):   {total_tokens:,}")
    click.echo(f"  storage:        {db_size_mb:.1f} MB  ({db_dir})")

    click.echo()
    click.echo("  ── by file type ──")
    click.echo(f"  {'type':<10s} {'files':>6s} {'chunks':>8s} {'chars':>12s}")
    for ft in sorted(type_counts, key=lambda t: -type_counts[t]):
        click.echo(f"  {ft:<10s} {type_counts[ft]:>6d} {type_chunks[ft]:>8d} {type_chars[ft]:>12,d}")

    if models_used:
        click.echo()
        click.echo("  ── embedding models ──")
        for m, cnt in models_used.most_common():
            click.echo(f"  {m}  ({cnt} files)")

    click.echo()
    click.echo("  ── contextual ──")
    click.echo(f"  done:           {n_ctx_done} / {n_files} files")
    if n_ctx_done > 0:
        ctx_files = [f for f in all_files.values() if f.get("contextual_done")]
        backends_used: Counter[str] = Counter()
        total_ctx_tokens = 0
        total_ctx_cost = 0.0
        for f in ctx_files:
            b = f.get("contextual_backend") or "unknown"
            backends_used[b] += 1
            total_ctx_tokens += f.get("context_tokens_used") or 0
            total_ctx_cost += f.get("context_cost_usd") or 0.0
        for b, cnt in backends_used.most_common():
            click.echo(f"  {b}:  {cnt} files")
        if total_ctx_tokens:
            click.echo(f"  tokens used:    {total_ctx_tokens:,}")
        if total_ctx_cost > 0:
            click.echo(f"  API cost:       ${total_ctx_cost:.4f}")
    if n_missing:
        click.echo(f"  source missing: {n_missing} files (source deleted from disk)")

    if files_with_timing:
        click.echo()
        click.echo("  ── ingest timing (cumulative) ──")
        click.echo(f"  extract:        {total_extract:.1f}s")
        click.echo(f"  embed:          {total_embed:.1f}s")
        click.echo(f"  upsert:         {total_upsert:.1f}s")

        # Per-type averages
        click.echo()
        click.echo("  ── avg time per file by type ──")
        click.echo(f"  {'type':<10s} {'extract':>8s} {'embed':>8s} {'upsert':>8s} {'files':>6s}")
        type_timing: dict[str, list] = defaultdict(list)
        for f in files_with_timing:
            ft = f.get("file_type") or "unknown"
            type_timing[ft].append(f)
        for ft in sorted(type_timing, key=lambda t: -len(type_timing[t])):
            files_t = type_timing[ft]
            n = len(files_t)
            avg_e = sum(f.get("extract_s") or 0 for f in files_t) / n
            avg_m = sum(f.get("embed_s") or 0 for f in files_t) / n
            avg_u = sum(f.get("upsert_s") or 0 for f in files_t) / n
            click.echo(f"  {ft:<10s} {avg_e:>7.2f}s {avg_m:>7.2f}s {avg_u:>7.2f}s {n:>6d}")

    # BM25 index
    from .ingestion.bm25_index import BM25_FILENAME
    bm25_path = db_dir / BM25_FILENAME
    if bm25_path.exists():
        bm25_mb = bm25_path.stat().st_size / 1024 / 1024
        click.echo()
        click.echo(f"  ── BM25 index ──")
        click.echo(f"  size:           {bm25_mb:.1f} MB")

    click.echo()
    meta_db.close()


# ---------------------------------------------------------------------------
# rebuild
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def rebuild(ctx: click.Context, yes: bool) -> None:
    """Drop ALL indexed data and re-ingest every document from scratch.

    Use this after:
    - Changing embedding.model_name in config.toml
    - Changing chunk_size or chunk_overlap
    - Suspecting index corruption

    The documents folder is not affected — only the chroma_db is cleared.
    """
    cfg: Config = ctx.obj["config"]
    if not yes:
        click.confirm(
            "This will DELETE all indexed data and re-ingest from scratch.\n"
            f"  DB: {cfg.paths.db_dir}\n"
            "Continue?",
            abort=True,
        )

    from .store.meta_db import MetaDB
    from .ingestion.bm25_index import BM25_FILENAME

    store = VectorStore(cfg.paths.db_dir, cfg.retrieval.collection_name)
    store._client.delete_collection(cfg.retrieval.collection_name)

    meta_db = MetaDB(cfg.paths.db_dir / "embd_meta.db")
    meta_db.reset()
    meta_db.close()

    bm25_path = cfg.paths.db_dir / BM25_FILENAME
    if bm25_path.exists():
        bm25_path.unlink()

    click.echo(f"Cleared collection '{cfg.retrieval.collection_name}'.")

    # Re-use the ingest command (inherits ctx.obj["config"])
    ctx.invoke(ingest)
