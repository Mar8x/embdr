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
              help="Run contextual generation on un-contextualized chunks after ingest.")
@click.option("--contextualize-estimate-only", is_flag=True,
              help="Print cost/time estimate for contextual generation, then exit.")
@click.pass_context
def ingest(
    ctx: click.Context,
    refresh: bool,
    reset: bool,
    contextualize: bool,
    contextualize_estimate_only: bool,
) -> None:
    """Scan the documents folder and ingest new or changed files.

    Files whose SHA-256 hash has not changed since the last ingest are
    skipped. Changed files are deleted from the store and re-indexed.
    Files that have been removed from disk are also cleaned up.

    A BM25 keyword index is built after every run that modifies the store.
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

    # --contextualize-estimate-only: print estimate and exit
    if contextualize_estimate_only:
        from .ingestion.contextual import estimate_contextualization, print_estimate
        estimate = estimate_contextualization(meta_db, cfg)
        print_estimate(estimate, cfg)
        meta_db.close()
        return

    # --reset: drop everything, prompt for confirmation
    if reset:
        if sys.stdin.isatty():
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

    encoder = Encoder(cfg.embedding.model_name, cfg.embedding.device)

    with Timer("total") as total_timer:
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

        # Delete stale data before re-ingesting changed files
        for path in scan.to_update:
            sk = _source_key(path, docs_dir)
            store.delete_file(sk)
            meta_db.reset_contextual(sk)

        # Remove files that no longer exist on disk
        for name in scan.to_remove:
            store.delete_file(name)
            meta_db.remove_file(name)

        to_process = scan.to_add + scan.to_update
        modified = bool(to_process or scan.to_remove)

        if not to_process and not scan.to_remove:
            click.echo("Nothing to ingest — all files are up to date.")
        else:
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
        if modified:
            click.echo("Building BM25 index ...")
            with Timer("bm25") as bm25_timer:
                build_and_save_bm25(store, cfg.paths.db_dir)
            click.echo(f"BM25 index built in {bm25_timer.elapsed:.1f}s")

    # --contextualize: run after ingest
    if contextualize or cfg.ingestion.contextual_ingestion:
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
@click.pass_context
def query(ctx: click.Context, question: str) -> None:
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
            hits = rrf_merge(
                semantic_hits, bm25_hits, store,
                semantic_weight=cfg.retrieval.semantic_weight,
                bm25_weight=cfg.retrieval.bm25_weight,
                top_k=cfg.retrieval.top_k,
            )
        else:
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
