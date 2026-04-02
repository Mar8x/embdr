# embd — Developer guide for Claude Code

## Project overview

embd is a RAG pipeline: ingest documents → embed with sentence-transformers → store in ChromaDB → query via CLI, TUI, or HTTP API. The HTTP API is designed as a ChatGPT Actions backend. Retrieval uses hybrid search (semantic + BM25 via Reciprocal Rank Fusion).

## Key commands

```bash
# Development
source .venv/bin/activate
embd ingest                            # ingest documents + build BM25 index
embd ingest --refresh                  # re-check all files by hash
embd ingest --reset                    # drop everything, full re-ingest
embd ingest --contextualize            # LLM context generation on chunks
embd ingest --contextualize-estimate-only  # cost/time estimate, no writes
embd query "question"                  # query from CLI (hybrid search)
embd shell                             # interactive TUI
embd serve                             # HTTP API

# Docker
docker compose up -d                        # dev (self-signed TLS)
docker compose -f docker-compose.yml \
  -f deploy/docker-compose.le.yml up -d     # production (Let's Encrypt)
```

## Project structure

- `src/embd/` — main Python package
  - `config.py` — loads `config.toml` + `.env`
  - `embedding/encoder.py` — sentence-transformers wrapper (MPS/CUDA/CPU)
  - `server.py` — FastAPI HTTP API
  - `ingestion/` — extractors, chunker, scanner, file watcher
    - `bm25_index.py` — BM25 index build/persist/query
    - `contextual.py` — contextual generation pipeline + cost estimation
  - `store/` — ChromaDB vector store + SQLite metadata
    - `vector_store.py` — ChromaDB wrapper
    - `meta_db.py` — SQLite file-level tracking, contextual progress
  - `qa/` — retrieval and generation
    - `retriever.py` — semantic + BM25 hybrid retriever
    - `hybrid_retriever.py` — RRF merge function
    - `generator_*.py` — MLX, Ollama, Claude backends
- `deploy/` — nginx reverse proxy, Docker, TLS, IP allowlist
- `config.toml` — all tunables (paths, models, chunking, retrieval weights, server)
- `.env` — secrets (git-ignored)
- `tests/` — pytest tests for meta_db, bm25, rrf, contextual estimation

## Conventions

- Config: `config.toml` for tunables, `.env` for secrets
- Chunk IDs are deterministic from `(source_key, page, chunk_index)`
- Embeddings are L2-normalized; ChromaDB uses cosine space
- File watcher (watchdog) auto-ingests on change during `serve`/`shell`
- BM25 index rebuilds after every ingest + 30s debounced rebuild in watcher
- SQLite `embd_meta.db` tracks file stats and contextual generation progress
- Contextual generation is resumable per-file

## Testing

```bash
pip install -e '.[all]' rank_bm25 pytest
pytest tests/
```
