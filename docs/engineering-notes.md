# Engineering Notes

---

## Contextual Ingestion: Inline vs Two-Pass

**Problem:** During ingest with contextualisation enabled, two models compete for GPU memory — the embedding model (BGE-M3 on MPS, ~2 GB) and the contextual LLM (Ollama/MLX, 2–5 GB). On Apple Silicon's unified memory, this causes throughput degradation for both.

**Observation:** PDF extraction is ~0.1s/file. Contextual LLM calls are ~2–6s/chunk (30–200 chunks/file). Re-parsing to avoid concurrent model loading costs <1% of total wall time.

### Options evaluated

| # | Strategy | Models in GPU | Re-parse? | Resumable? | Verdict |
|---|----------|--------------|-----------|------------|---------|
| A | **Inline** — contextualize during ingest, before embedding | LLM + embedding simultaneously | No | Per-file (partial files lost on interrupt) | Memory contention; MLX measured slower than Ollama due to shared Metal context |
| B | **Two-pass** — ingest all raw, then contextualize + re-embed | One at a time | Yes (~0.1s/file) | Per-file via SQLite `contextual_done` flag | Best throughput; each model gets full GPU; negligible re-parse cost |
| C | **Three-pass** — extract all → contextualize all → embed all in bulk | One at a time | Yes | Needs intermediate state on disk | Marginal gain over B; adds complexity (temp storage, interrupted state); embedding batches already efficient in B |
| D | **Same model for both** — use a single model that embeds + generates context | One model | No | — | No model exists that does both well; embedding models can't generate; LLMs produce poor embeddings |

### Decision: Two-pass (B) as default

- **Pass 1** `embd ingest` — extract → chunk → embed raw text → store in ChromaDB. Only embedding model loaded.
- **Pass 2** `contextualize_files()` — iterate un-contextualized files, re-parse from disk, LLM per chunk, re-embed, update ChromaDB. Embedding model reloads but LLM gets full GPU during generation phase.

**Why not inline (A)?**
Measured: MLX contextual + BGE-M3 embedding concurrent = slower than Ollama sequential. Even with Ollama (separate process), there's Metal shader contention. The 0.1s/file re-parse penalty is irrelevant against minutes of LLM work per file.

**Why not three-pass (C)?**
Marginal improvement. Pass B already sequences model usage naturally (all LLM calls → all re-embeds per file). Bulk embedding across files (C) saves ~10% on embedding but requires holding all contextualized texts in temp storage and complicates interrupt/resume.

**Ollama vs MLX for contextual generation:**
Ollama runs in a separate process → no Python GIL or Metal context sharing with the embedding model. MLX loads into the same process → direct GPU resource competition. For bulk ingestion where the embedding model is also active, Ollama with a small model (`qwen3:4b`, 2.5 GB) outperformed MLX with `Llama-3.2-3B-Instruct-4bit` in practice.

MLX may be faster for retroactive contextualisation (no embedding model loaded), but the difference is small enough that Ollama is the safer default.

---

## Contextual Generation: Model Selection

Measured on Apple M4 Pro using `embd ingest --recontext-file` on a representative document.

### Benchmark results

| Model | Backend | Throughput |
|---|---|---|
| qwen3:1.7b | Ollama | 79 tok/s |
| Llama-3.2-1B-Instruct-4bit | MLX | 62 tok/s |
| Qwen2.5-1.5B-Instruct-4bit | MLX | 35 tok/s |
| qwen3:4b | Ollama | 38 tok/s |
| Llama-3.2-3B-Instruct-4bit | MLX | 16 tok/s |

### Finding

**`qwen3:4b` is the recommended minimum.** Despite being 2× slower than `qwen3:1.7b`, the quality gap matters for this task.

Context generation requires strict instruction adherence — the prompt asks for a short, on-topic situating sentence and nothing else. At 1–1.7B parameters, models drift: they ramble, miss the format constraint, or produce generic summaries that don't accurately situate the chunk. **A bad context string is worse than none** — it corrupts the embedding and degrades retrieval for that chunk until explicitly re-contextualized.

The Qwen3 scaling report confirms the gap is non-linear: Qwen3-4B exceeds Qwen2.5-7B on reasoning and instruction-following, while Qwen3-1.7B tracks closer to Qwen2.5-3B. This translates directly to structured-output reliability at short output lengths.

Sub-1B models (Llama-3.2-1B) are not suitable — throughput gain over 1.7B is marginal and output quality degrades further.

**Sources:**
- Qwen3 Technical Report — https://arxiv.org/pdf/2505.09388
- Qwen3 Blog (Alibaba) — https://qwenlm.github.io/blog/qwen3/
- Qwen3 in RAG pipelines — https://medium.com/@marketing_novita.ai/qwen-3-in-rag-pipelines-all-in-one-llm-embedding-and-reranking-solution-619fe1acfe11
