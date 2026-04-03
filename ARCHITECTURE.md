# embd — Architecture

## System diagram

![Architecture overview](docs/architecture.png)

## Overview

```
documents/  →  Ingest (extract + chunk + embed)  →  ChromaDB + BM25 index
                                                          ↓
                    (optional) Contextual generation  →  re-embed chunks
                                                          ↓
CLI / TUI / HTTP API  →  Query (embed + hybrid search)  →  LLM generates answer
```

## Ingest flow

1. **Scan** — recursively walk `documents_dir`, hash each file, skip unchanged. Track file stats in SQLite (`embd_meta.db`).
2. **Extract** — dispatch by type: PyMuPDF (PDF), ebooklib (EPUB), python-docx (DOCX), openpyxl (XLSX), plain read (TXT/MD), httpx+BS4 (URLs).
3. **Chunk** — sentence-aware sliding window (`chunk_size` / `chunk_overlap`).
4. **Embed** — sentence-transformers (e.g. BGE-M3) on MPS/CUDA/CPU.
5. **Store** — upsert into ChromaDB (cosine space, HNSW index).
6. **BM25** — build keyword index from all chunk texts, persist as `bm25_index.pkl` (atomic write).

Chunk IDs are deterministic from `(source_key, page, chunk_index)` — re-ingesting unchanged files is a no-op.

### Contextual generation (optional)

When `--contextualize` is passed, contextualisation runs as a **second pass** after all embedding is complete. This ensures only one model uses the GPU at a time (embedding model during pass 1, LLM during pass 2).

1. **Pass 1** completes normally — all chunks are embedded and searchable via semantic + BM25 hybrid search.
2. **Pass 2** iterates un-contextualized files: re-parses each from disk (~0.1s/file, negligible), sends each chunk + full document text to an LLM, prepends the returned ~75-token context string, re-embeds, and updates ChromaDB in place.

Models for context generation are configured independently in `[ingestion.contextual]` — you can use a small/cheap model (e.g., Haiku, qwen3:4b) for context while using a larger model for answering in `[llm]`.

For documents exceeding `max_doc_tokens`, a sliding window of surrounding chunks replaces the full document text. Progress is tracked per-file in SQLite — interrupted runs resume automatically.

Claude backend uses `cache_control: ephemeral` for prompt caching, so repeated calls for chunks from the same document hit the cache at 1/10th input cost.

## Query flow

1. **Embed** the question with the same model.
2. **Semantic search** — ChromaDB HNSW cosine search for top candidates.
3. **BM25 search** — keyword match against the BM25 index (when available).
4. **Merge** — Reciprocal Rank Fusion (RRF) combines both ranked lists. Formula: `score(d) = sum(weight / (k + rank + 1))` across both lists, with `k=60`.
5. **Generate** an answer with the configured LLM (MLX, Ollama, or Claude) using retrieved passages as context.

When no BM25 index exists (e.g. first query before any ingest), falls back to pure semantic search.

## Key modules

| Module | Responsibility |
|--------|---------------|
| `src/embd/embedding/encoder.py` | Loads sentence-transformers, auto-selects device |
| `src/embd/store/vector_store.py` | ChromaDB wrapper — upsert, query, delete |
| `src/embd/store/meta_db.py` | SQLite file-level tracking, contextual progress, benchmarks |
| `src/embd/ingestion/` | Extractors, chunker, scanner, file watcher |
| `src/embd/ingestion/bm25_index.py` | BM25 index build, persist (atomic pickle), query |
| `src/embd/ingestion/contextual.py` | Contextual generation pipeline + cost estimation |
| `src/embd/qa/hybrid_retriever.py` | Reciprocal Rank Fusion merge |
| `src/embd/qa/retriever.py` | Combined semantic + BM25 retriever |
| `src/embd/qa/` | MLX, Ollama, Claude generation backends |
| `src/embd/server.py` | FastAPI HTTP API (OpenAI retrieval-plugin compatible) |
| `src/embd/config.py` | Loads `config.toml` + `.env` |

## Storage

### ChromaDB (`chroma_db/`)
- `chroma.sqlite3` — metadata, chunk text, IDs
- `<segment-uuid>/` — HNSW index files (vectors, graph links, norms)

Metadata per chunk: `source_filename`, `page_num`, `chunk_index`, `file_hash`, `embedding_model`, `ingestion_timestamp`, `document_date`, `document_date_quality`.

### BM25 index (`bm25_index.pkl`)
- Pickled `BM25Okapi` + chunk ID mapping
- Rebuilt after every ingest that modifies the store
- Atomic write: `.pkl.tmp` → `os.replace()` → `.pkl`

### SQLite metadata (`embd_meta.db`)

**`files` table** — one row per ingested file:
- `source_key`, `file_hash`, `mtime`, `char_count`, `token_count`, `chunk_count`
- Contextual tracking: `contextual_done`, `contextual_at`, `contextual_backend`, `context_tokens_used`, `context_cost_usd`, `used_sliding_window`

**`ollama_benchmarks` table** — throughput measurements per model + machine:
- `model`, `machine_id`, `tok_per_sec`, `sample_chunks`, `measured_at`

Used for cost/time estimation (`--contextualize-estimate-only`) and resumable contextual generation.
