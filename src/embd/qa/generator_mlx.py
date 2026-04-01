"""
LLM answer generator using Apple's MLX framework (default backend).

MLX runs natively on Apple Silicon's unified memory architecture:
- No CPU↔GPU memory copies — the model weights live in the shared pool
- Metal shader compilation happens once on first generate() call
- Lazy loading via cached_property keeps CLI startup instant

The grounded-answer prompt structure:
1. System prompt: establishes "closed book" constraint and citation format once
2. Context section: one labelled block per retrieved chunk, source embedded inline
3. User message: the question + instruction to cite

Inline source tags (e.g. [report.pdf, p.4]) are embedded in the context
blocks themselves, not appended at the end, so the model can associate
claims with sources as it reads rather than having to look them up.
"""
from __future__ import annotations

import logging
from functools import cached_property

from .retriever import RetrievedChunk
from .token_usage import TokenUsage

logger = logging.getLogger(__name__)

# Default "grounded" system prompt — strict closed-book Q&A from passages only.
SYSTEM_PROMPT_GROUNDED = """\
You are a precise document assistant. Your ONLY job is to answer questions \
using the text passages provided below.

Rules:
1. Base your answer EXCLUSIVELY on the provided passages.
2. After each claim, cite the source in brackets:
   - For local documents: [filename, p.N]
   - For web sources: [title or domain, URL]
3. If the passages do not contain sufficient information to answer, respond with exactly:
   "I don't know based on the provided documents."
4. Do not use prior knowledge, make assumptions, or invent information beyond what is stated.
5. Be concise and factual.
6. Always include the full URL when citing a web source so the user can visit it.
7. Do not use markdown tables (hard to read in a terminal). Prefer short paragraphs or bullet lists.
8. Passage headers may include a Date and quality note. Use this to judge recency: \
prefer newer sources when passages conflict, and note when information may be outdated."""

# Research mode: document-grounded but allows synthesis and general knowledge where labeled.
# Pairs well with Claude and /search (SearXNG snippets in the user message).
SYSTEM_PROMPT_RESEARCH = """\
You are a research assistant. You receive excerpts from the user's indexed \
documents in the context below; the user's question may also include web search \
results (titles, URLs, snippets) when available.

Use the provided passages and any web snippets as your primary evidence. You may \
supplement with careful general knowledge or reasoning to connect ideas, explain \
context, or suggest what to verify next—clearly distinguish what comes directly \
from the sources versus what is synthesis or background knowledge.

Rules:
1. Cite local documents as [filename, p.N] whenever you use them.
2. Cite web material with [title or domain, full URL] when present in the context.
3. Label clearly when you are inferring, generalizing, or using outside knowledge.
4. Do not invent specific facts, figures, or quotes that are not supported by the context.
5. If the context is insufficient, say what is missing and what could be researched next.
6. Do not use markdown tables (hard to read in a terminal). Prefer short paragraphs or bullet lists.
7. Passage headers may include a Date and quality note. Use this to judge recency and \
flag when information may be outdated or when newer sources should take precedence."""

# Backward-compatible name for imports expecting SYSTEM_PROMPT.
SYSTEM_PROMPT = SYSTEM_PROMPT_GROUNDED


def resolve_system_prompt(
    preset: str,
    override: str | None,
) -> str:
    """Return the system prompt string from config preset and optional full override."""
    if override is not None and str(override).strip():
        return str(override).strip()
    p = (preset or "grounded").strip().lower()
    if p == "research":
        return SYSTEM_PROMPT_RESEARCH
    return SYSTEM_PROMPT_GROUNDED


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as labelled passages with inline attribution."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        label = f"Source: {c.source_filename}, Page {c.page_num}"
        if c.document_date:
            date_brief = c.document_date[:10]
            qual = {"document_metadata": "from document", "filesystem": "file mtime"}
            qual_note = qual.get(c.document_date_quality, "")
            label += f", Date {date_brief}"
            if qual_note:
                label += f" ({qual_note})"
        parts.append(f"--- Passage {i} [{label}] ---\n{c.text}")
    return "\n\n".join(parts)


def _apply_template(tokenizer, messages: list[dict], model_path: str) -> str:
    """Format messages using the tokenizer's chat template.

    Some models (e.g. Mistral) require strict user/assistant alternation and
    reject a leading 'system' role. When that happens we retry with the system
    prompt merged into the first user message — a universally safe fallback.
    """
    if not (hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template):
        logger.warning(
            "Tokenizer for '%s' has no chat_template; falling back to raw concat.",
            model_path,
        )
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        return f"{system}\n\n{user}" if system else user

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Model doesn't support a system role (e.g. Mistral v0.3).
        # Merge system prompt into the first user turn and retry.
        logger.debug(
            "Chat template rejected system role for '%s'; merging into user turn.",
            model_path,
        )
        merged = [m for m in messages if m["role"] != "system"]
        system_content = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        if system_content and merged:
            merged[0] = {
                "role": "user",
                "content": f"{system_content}\n\n{merged[0]['content']}",
            }
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True
        )


def _build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    context = _build_context(chunks)
    return f"Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer (cite sources):"


def upstream_prompt_text(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    backend: str = "mlx",
    system_prompt: str | None = None,
) -> str:
    """Human-readable preview of messages sent to the LLM (Claude / Ollama / MLX).

    Claude and Ollama use ``system`` + ``user`` roles exactly as shown.
    MLX applies ``tokenizer.apply_chat_template`` to the same logical messages
    before generation, so special tokens may wrap this content.
    """
    user = _build_user_message(question, chunks)
    warn = ""
    if not chunks:
        warn = (
            "NOTE: With zero retrieved passages, `generate()` returns a canned "
            "fallback and does not call the model — except your question string "
            "may still carry web-search context in /search flows.\n\n"
        )
    note = ""
    b = backend.strip().lower()
    if b == "mlx":
        note = (
            "Note: MLX applies the model's chat template (e.g. [INST]…[/INST]) "
            "to the system+user messages below before running generation.\n\n"
        )
    elif b == "claude":
        note = (
            "Note: Sent via Anthropic Messages API as "
            "system=SYSTEM, user=USER below.\n\n"
        )
    elif b == "ollama":
        note = "Note: Sent as chat messages with roles system and user.\n\n"
    sp = system_prompt if system_prompt is not None else SYSTEM_PROMPT_GROUNDED
    return (
        f"{warn}{note}"
        f"=== SYSTEM ===\n\n{sp}\n\n"
        f"=== USER ===\n\n{user}\n"
    )


class MLXGenerator:
    """Generate answers with an MLX model loaded into Apple Silicon unified memory."""

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> None:
        self._model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt = system_prompt or SYSTEM_PROMPT_GROUNDED

    @cached_property
    def _loaded(self) -> tuple:
        """Lazy-load model and tokenizer on first generate() call.

        Importing mlx_lm here (not at module level) means this file can be
        imported even if mlx-lm is not installed — e.g. when running with
        the Ollama backend.
        """
        from mlx_lm import load  # type: ignore[import-untyped]

        logger.info("Loading MLX model '%s' into unified memory …", self._model_path)
        model, tokenizer = load(self._model_path)
        logger.info("MLX model loaded and ready.")
        return model, tokenizer

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, TokenUsage | None]:
        """Generate a grounded answer.

        Returns (answer_text, token usage for the model call).
        """
        from mlx_lm import generate as mlx_generate  # type: ignore[import-untyped]

        if not chunks:
            return "I don't know based on the provided documents.", None

        model, tokenizer = self._loaded
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": _build_user_message(question, chunks)},
        ]

        # apply_chat_template formats the conversation with the model's own
        # special tokens ([INST], <|im_start|>, etc.). Without this, the model
        # may not recognize it as an instruction-following task.
        prompt = _apply_template(tokenizer, messages, self._model_path)

        # mlx-lm >= 0.21 uses a sampler callable instead of a bare temp= kwarg.
        # make_sampler() returns a callable that generate_step() accepts.
        # verbose=False suppresses mlx-lm's own timing output; we display
        # our own via perf.py for a consistent UX across backends.
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-untyped]

        output: str = mlx_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temperature),
            verbose=False,
        )

        usage: TokenUsage | None = None
        try:
            n_in = len(tokenizer.encode(prompt))
            n_out = len(tokenizer.encode(output))
            usage = TokenUsage(prompt_tokens=n_in, completion_tokens=n_out)
        except Exception:
            pass

        return output.strip(), usage
