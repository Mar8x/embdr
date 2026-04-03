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

Measured on Apple M4 Pro using `embd ingest --recontext-file`.

### Measurement methodology note

Initial benchmarks used **wall time / output tokens**, which included LLM prefill time. For large input contexts (sliding window or full-doc), prefill dominated and deflated the numbers. As of 2026-04-03, the Ollama backend was fixed to use `eval_duration` from the Ollama response (generation time only), giving accurate generation tok/s regardless of input size. MLX still uses wall time (no separate metric available).

### Benchmark results (corrected, 2026-04-03)

| Model | Backend | Throughput | Notes |
|---|---|---|---|
| qwen3:1.7b | Ollama | 104 tok/s | Fastest; quality risk at 1.7B |
| qwen3:4b | Ollama | 62 tok/s | Best quality/speed balance |
| Llama-3.2-1B-Instruct-4bit | MLX | 36 tok/s | MLX native; quality marginal |
| Qwen2.5-1.5B-Instruct-4bit | MLX | 18 tok/s | MLX native; better than Llama-1B |
| Llama-3.2-3B-Instruct-4bit | MLX | 9 tok/s | Slow on MPS; use Ollama instead |

Previous measurements (wall time, inflated by prefill): qwen3:1.7b 79, Llama-1B 62, Qwen2.5-1.5B 35, qwen3:4b 38, Llama-3B 16.

### Hardware choice: Apple M4 Pro

Active configuration: **MLX backend, `Qwen2.5-1.5B-Instruct-4bit`** (18 tok/s).

Chosen over faster alternatives for two reasons:
- **No Ollama dependency** — MLX runs natively in-process, no server to manage. Useful when contextualizing in isolation without Ollama running.
- **Better instruction following than Llama-1B** — Qwen2.5-1.5B produces more consistent structured output than Llama-3.2-1B despite lower throughput. The quality gap vs Llama-1B justifies the 2× speed cost.

If throughput is the priority and Ollama is available, **qwen3:4b via Ollama at 62 tok/s** is the best overall choice: strong instruction following, 3× faster than Qwen2.5-1.5B MLX.

Quality recommendation unchanged: **avoid sub-1.5B models** — they drift from the format constraint and produce generic summaries that corrupt embeddings.

**Sources:**
- Qwen3 Technical Report — https://arxiv.org/pdf/2505.09388
- Qwen3 Blog (Alibaba) — https://qwenlm.github.io/blog/qwen3/
- Qwen3 in RAG pipelines — https://medium.com/@marketing_novita.ai/qwen-3-in-rag-pipelines-all-in-one-llm-embedding-and-reranking-solution-619fe1acfe11

---

## Contextual Generation: Full-Doc vs Sliding Window Performance

### Finding: set `max_doc_tokens` low — full-doc mode is an Ollama/MLX trap

With `max_doc_tokens = 50000`, documents below 50,000 tokens use **full-doc mode**: the entire document text is sent as context on every single chunk call. Ollama and MLX do **not** cache the KV state between calls, so each call re-processes the full document from scratch.

Observed on Apple M4 Pro with qwen3:4b (Ollama):

| Mode | Input tokens/call | Observed tok/s |
|---|---|---|
| Full-doc (19,000 token doc) | ~19,000 | ~1 tok/s |
| Sliding window (window=3) | ~3,500 | ~38 tok/s |

At ~500 tok/s prefill speed, a 19,000-token document costs 38 seconds of prefill per chunk — completely dominating the ~2-second generation phase. The measured tok/s (output tokens / total wall time) collapses to ~1–2 tok/s.

**Fix:** set `max_doc_tokens = 4000` (~16,000 chars). This pushes virtually all real-world documents into sliding window mode, capping input per call at `(2×window_chunks+1) × chunk_size` chars (~14,000 chars / ~3,500 tokens). Throughput returns to the expected 30–40 tok/s.

**Claude is unaffected** — prompt caching (`cache_control: ephemeral`) makes repeated full-doc calls cheap: the document is cached after the first chunk and subsequent calls hit the cache at 1/10th input cost. The `max_doc_tokens` threshold matters only for Ollama and MLX.
