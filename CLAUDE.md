# embd — Developer guide for Claude Code

## Project overview

embd is a RAG pipeline: ingest documents → embed with sentence-transformers → store in ChromaDB → query via CLI, TUI, or HTTP API. The HTTP API is designed as a ChatGPT Actions backend.

## Key commands

```bash
# Development
source .venv/bin/activate
embd ingest              # ingest documents
embd query "question"    # query from CLI
embd shell               # interactive TUI
embd serve               # HTTP API

# Docker
docker compose up -d                        # dev (self-signed TLS)
docker compose -f docker-compose.yml \
  -f deploy/docker-compose.le.yml up -d     # production (Let's Encrypt)
```

## Project structure

- `src/embd/` — main Python package
  - `config.py` — loads `config.toml` + `.env`
  - `encoder.py` — sentence-transformers wrapper (MPS/CUDA/CPU)
  - `server.py` — FastAPI HTTP API
  - `ingestion/` — extractors, chunker, scanner, file watcher
  - `store/` — ChromaDB vector store
  - `llm/` — MLX, Ollama, Claude backends
- `deploy/` — nginx reverse proxy, Docker, TLS, IP allowlist
- `config.toml` — all tunables (paths, models, chunking, server)
- `.env` — secrets (git-ignored)

## Conventions

- Config: `config.toml` for tunables, `.env` for secrets
- Chunk IDs are deterministic from `(source_key, page, chunk_index)`
- Embeddings are L2-normalized; ChromaDB uses cosine space
- File watcher (watchdog) auto-ingests on change during `serve`/`shell`

## Testing

```bash
pip install -e '.[dev]'
pytest
```
