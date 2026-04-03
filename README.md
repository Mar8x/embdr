# embd — Turn your documents into a ChatGPT GPT

[![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet?logo=anthropic)](https://claude.ai/code)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChatGPT Actions](https://img.shields.io/badge/ChatGPT-Actions-74aa9c?logo=openai&logoColor=white)](https://platform.openai.com/docs/actions)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-embeddings-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/BAAI/bge-m3)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Index your documents (PDF, EPUB, Markdown, DOCX, XLSX, plain text) and expose them as a retrieval API for a ChatGPT custom GPT. Ask your GPT questions — get grounded, cited answers from your own files.

Runs on **Apple Silicon** (MPS + MLX), **Linux/Docker** (CPU or CUDA), or any platform with the **Claude API**.

## How it works

1. **Ingest** — scans your `documents/` folder, extracts text, chunks it, and stores embeddings in ChromaDB. A BM25 keyword index is built alongside for hybrid search.
2. **Serve** — exposes an OpenAI-compatible HTTP API (`POST /query`) with hybrid retrieval (semantic + BM25 via Reciprocal Rank Fusion).
3. **GPT** — your ChatGPT custom GPT calls the API as an Action and answers from your documents.

You can also query locally via CLI (`embd query "..."`) or an interactive TUI (`embd shell`).

See [ARCHITECTURE.md](ARCHITECTURE.md) for system diagrams, module details, and storage internals.

---

## Quickest path: local install + tunnel

The simplest way to get a working GPT. No VPS, no nginx, no certificates to manage. ChatGPT Actions requires **HTTPS on port 443 with a valid TLS certificate** — a tunnel service handles that for you.

### 1. Install and ingest

```bash
git clone https://github.com/Mar8x/embdr.git && cd embdr
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'

cp .env.example .env
# edit .env — set EMBD_API_KEY (generate: openssl rand -hex 32)

# Drop your files into documents/
embd ingest
```

### 2. Start the API

```bash
embd serve
# Listening on http://0.0.0.0:8000
```

### 3. Expose via tunnel

Use [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) (free, recommended) or [ngrok](https://ngrok.com/):

```bash
# Cloudflare Tunnel (install cloudflared first):
cloudflared tunnel --url http://localhost:8000

# Or ngrok:
ngrok http 8000
```

This gives you a public `https://....` URL on port 443 with valid TLS — exactly what ChatGPT needs.

### 4. Create your GPT

1. In ChatGPT, create a new GPT.
2. Under **Actions**, import `https://<your-tunnel-url>/openapi.json`.
3. Authentication: **API Key**, type **Bearer**, value = your `EMBD_API_KEY`.
4. Paste the instructions from [docs/chatgpt-gpt-instructions.md](docs/chatgpt-gpt-instructions.md) into the GPT's **Instructions** field.

Done. Your GPT now answers from your documents with inline citations.

---

## Production: Docker Compose + nginx

For a permanent deployment on a VPS with your own domain.

Docker Compose runs **nginx as a reverse proxy** in front of `embd serve`:

- **IP whitelist** — only OpenAI's ChatGPT servers and your custom IPs are allowed.
- **Bearer token** — verified in nginx before proxying.
- **TLS termination** — Let's Encrypt or self-signed for dev.

```bash
# 1. Set secrets
cp .env.example .env
# edit .env — set EMBD_API_KEY, ANTHROPIC_API_KEY, EMBD_PUBLIC_HOST, etc.

# 2. Seed the OpenAI IP allowlist
./deploy/update-openai-ips.sh

# 3. Add your own IPs (optional)
# Edit deploy/custom-ips.conf — one CIDR per line

# 4. Launch
# Dev (self-signed TLS):
docker compose up -d

# Production (Let's Encrypt on host):
# Set EMBD_TLS_FULLCHAIN, EMBD_TLS_PRIVKEY, EMBD_LISTEN_PORT=443 in .env
docker compose -f docker-compose.yml -f deploy/docker-compose.le.yml up -d
```

The nginx container refreshes OpenAI's IP list daily at 03:00 UTC.

### Standalone Docker (without nginx)

```bash
docker build -t embd .
docker run -d --name embd \
  -p 8000:8000 \
  -v ./documents:/app/documents \
  -v ./chroma_db:/app/chroma_db \
  -v ./config.toml:/app/config.toml:ro \
  -e EMBD_API_KEY="your-key" \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  embd
```

Put a reverse proxy or tunnel in front for HTTPS on 443.

### Server management (`embd-stack.sh`)

A helper script wraps Docker Compose for common production operations:

```bash
./scripts/ops/embd-stack.sh start          # docker compose up -d
./scripts/ops/embd-stack.sh stop           # docker compose stop
./scripts/ops/embd-stack.sh deploy         # git pull + build + up -d
./scripts/ops/embd-stack.sh deploy-embd    # rebuild embd only
./scripts/ops/embd-stack.sh deploy-nginx   # rebuild nginx only
./scripts/ops/embd-stack.sh logs           # follow logs
./scripts/ops/embd-stack.sh ps             # container status
```

### Syncing embeddings to a server

A common setup: ingest documents on a powerful machine (e.g. Apple Silicon with MPS acceleration) and serve from a lightweight CPU VPS. The rsync helper syncs your local `chroma_db/` to the remote server:

```bash
# Dry run first
REMOTE=user@server REMOTE_ROOT=/path/to/embdr \
  ./scripts/ops/rsync-chroma-to-server.sh --dry-run

# Sync for real — optionally stop embd on the remote during transfer
REMOTE=user@server REMOTE_ROOT=/path/to/embdr \
  ./scripts/ops/rsync-chroma-to-server.sh --stop-embd-first
```

This lets you keep ingestion fast (GPU-accelerated embeddings) while the serving VPS only needs CPU for vector search.

---

## CLI commands

| Command | Description |
|---------|-------------|
| `embd ingest` | Embed new/changed files (search works after this) |
| `embd ingest --contextualize` | Embed, then contextualize all un-contextualized files |
| `embd ingest --recontext` | Re-contextualize ALL files from scratch |
| `embd ingest --recontext-file KEY` | Re-contextualize a single file by source key |
| `embd ingest --contextualize-estimate-only` | Print cost/time estimate, no writes |
| `embd ingest --refresh` | Re-check all files by hash, skip unchanged |
| `embd ingest --reset` | Drop everything and re-ingest from scratch |
| `embd ingest -y` | Skip confirmation prompt |
| `embd ingest-url <url>` | Fetch a web page and ingest it |
| `embd query "question"` | Ask a question against indexed documents |
| `embd query "question" --diag` | Same, with retrieval diagnostics (RRF scores) |
| `embd stats` | Show index statistics, timing, contextual status |
| `embd shell` | Interactive TUI for Q&A |
| `embd serve` | HTTP retrieval API |
| `embd delete <source_key>` | Remove one file's chunks |
| `embd rebuild` | Drop everything and re-ingest from scratch |

Flags compose: `--reset` implies `--refresh`. Combine freely: `embd ingest --refresh --contextualize -y`.

## Supported file types

| Type | Extensions | Optional dep |
|------|-----------|-------------|
| `pdf` | `.pdf` | — (core; OCR with `embd[ocr]`) |
| `epub` | `.epub` | — (core) |
| `txt` | `.txt`, `.text`, `.rst`, `.log` | — (core) |
| `md` | `.md`, `.markdown` | — (core) |
| `docx` | `.docx` | `pip install 'embd[docx]'` |
| `xlsx` | `.xlsx` | `pip install 'embd[xlsx]'` |

## Configuration

Edit `config.toml`:

- **`[paths]`** — `documents_dir`, `db_dir`
- **`[ingestion]`** — `chunk_size`, `chunk_overlap`, `enabled_types`, `ignore_patterns`, OCR settings
- **`[ingestion.contextual]`** — `backend`, `ollama_model`, `claude_model`, `mlx_model`, `max_doc_tokens`
- **`[embedding]`** — `model_name` (default: `BAAI/bge-m3`), `device` (`auto`/`mps`/`cuda`/`cpu`)
- **`[retrieval]`** — `top_k`, `collection_name`, `semantic_weight`, `bm25_weight`
- **`[llm]`** — `backend` (`mlx`/`ollama`/`claude`), model settings
- **`[server]`** — `host`, `port` for `embd serve`

After changing the embedding model or chunk parameters, run `embd ingest --reset`.

### Hybrid search (BM25 + semantic)

Every `embd ingest` run builds a BM25 keyword index alongside the vector embeddings. At query time, results from both are merged via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (RRF). This catches exact matches (product names, clause numbers, acronyms) that pure semantic search misses.

Tune the blend in `config.toml` under `[retrieval]`:

```toml
semantic_weight = 0.8   # vector cosine similarity
bm25_weight     = 0.2   # keyword TF-IDF
```

Set `bm25_weight = 0` to disable BM25 entirely. The BM25 index is always built (cheap, local, no API calls) so you can toggle it without re-ingesting.

### Contextual generation

Optionally prepend a short LLM-generated context string to each chunk before embedding. This situates the chunk within its document, improving retrieval quality for ambiguous passages.

When `--contextualize` is passed, ingestion runs in two passes so only one model uses the GPU at a time:

1. **`embd ingest`** — extract, chunk, and embed new/changed files. Semantic search, BM25, and hybrid search all work immediately after this.
2. **`embd ingest --contextualize`** — same as above, then contextualize **all un-contextualized files** (not just the newly ingested ones). Files that were previously ingested without `--contextualize` are included. Already-contextualized files are always skipped.

You can embed first and contextualize later — or do both in one command:

```bash
# Embed only (fast, search works after this)
embd ingest

# Contextualize everything that hasn't been contextualized yet
embd ingest --contextualize

# See what it would cost before committing
embd ingest --contextualize-estimate-only

# Non-interactive (scripts / cron)
embd ingest --contextualize -y
```

Contextual generation is resumable — interrupted runs pick up where they left off.

#### Calibrating the time estimate (Ollama / MLX)

`--contextualize-estimate-only` uses a measured tok/s value for your model and machine. On the first run it falls back to a conservative default (60 tok/s), which may be way off. Run a quick calibration first with one representative file — a document of a few hundred KB, 20–100 chunks:

```bash
# Process one file to measure real throughput (stored in SQLite, ~1–2 min)
embd ingest --recontext-file papers/some-report.pdf

# Now the estimate uses your actual tok/s
embd ingest --contextualize-estimate-only
```

The measured tok/s is stored per-model in `chroma_db/embd_meta.db` and persists across runs. Switching to a different model in `config.toml` resets to the default until you calibrate the new model.

Three backends are supported:

- **Claude** (`backend = "claude"`) — fast, cheap with prompt caching. Requires `ANTHROPIC_API_KEY`.
- **Ollama** (`backend = "ollama"`) — local, free, no API cost. Recommended for bulk ingestion.
- **MLX** (`backend = "mlx"`) — Apple Silicon only, native Metal. Requires `pip install 'embd[mlx]'`.

For empirical throughput measurements and model quality trade-offs, see [docs/engineering-notes.md](docs/engineering-notes.md).

Models for context generation are configured independently from the inference models in `[llm]`, so you can use a small/cheap model for contextualization and a larger model for answering. Configure in `config.toml` under `[ingestion.contextual]`:

```toml
[ingestion.contextual]
backend       = "ollama"                       # "claude", "ollama", or "mlx"
ollama_model  = "qwen3:4b"                    # small/fast, independent from [llm]
claude_model  = "claude-haiku-4-5-20251001"   # cheap — good for context strings
mlx_model     = "mlx-community/Llama-3.2-3B-Instruct-4bit"
max_doc_tokens = 50000                         # above this, use sliding window
window_chunks  = 3                             # N chunks before + N after
```

Contextual generation is resumable — interrupted runs pick up where they left off.

### OCR

Embedded images in PDFs and DOCX files can be OCR'd during ingestion. Install `embd[ocr]` for Surya (high-quality, MPS/CUDA) or `embd[ocr-cpu]` + Tesseract for CPU/Docker. The `ocr_embedded_images` setting in `config.toml` controls when OCR runs: `"when_no_text"` (default — only scanned pages), `"always"`, or `"never"`.

### Web search (SearXNG)

Optional web search is available in `embd query` and `embd shell` (CLI/TUI) via a SearXNG instance. It is **not** used by `embd serve` or the ChatGPT Actions API. Configure the URL in the `[search]` section of `config.toml`.

### LLM backends

- **MLX** (Apple Silicon only): `pip install 'embd[mlx]'`. Models download on first use.
- **Ollama** (any platform): `ollama serve`, then `ollama pull <model>`.
- **Claude** (any platform): set `ANTHROPIC_API_KEY` in `.env`.

### Secrets (.env)

| Variable | Required when | Notes |
|----------|--------------|-------|
| `ANTHROPIC_API_KEY` | `backend = "claude"` | Anthropic API key |
| `EMBD_API_KEY` | `embd serve` | Bearer token for the retrieval API |
| `HF_TOKEN` | Never (optional) | Silences HuggingFace warnings |

## Retrieval API

Retrieval uses hybrid search (semantic + BM25 via RRF) with no reranking step. If recall feels low, increase `top_k` or adjust `semantic_weight`/`bm25_weight` in `config.toml`.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/query` | Bearer | Semantic search over the index |
| GET | `/health` | None | Liveness check |
| GET | `/openapi.json` | None | OpenAPI schema (import into ChatGPT) |
| GET | `/docs` | None | Swagger UI |

## Data privacy

Your documents never leave your machine — embeddings, BM25 indexing, and vector search all run locally. The only external calls are: (1) ChatGPT receives retrieved text snippets as part of each query, and (2) `--contextualize` with the Claude backend sends chunk text to the Anthropic API (opt-in, only when you pass `--contextualize`). On personal ChatGPT plans, standard OpenAI data terms apply to those snippets; ChatGPT Team and Enterprise plans have stronger data handling commitments and no training on your data. Choose what you index accordingly, and review OpenAI's current data policies for your plan before indexing sensitive material.

## License

MIT
