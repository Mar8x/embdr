"""
Interactive Q&A shell built with Textual.

Output is rendered in a read-only TextArea so users get a real cursor,
shift+arrow text selection, and Ctrl+C to copy — just like the input box.
Vim-style keys (j/k/gg/G/yy/ys) work in NORMAL mode (press Esc).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
import logging

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Footer, Header, Input, Static, TextArea

from .config import Config
from .display_format import format_local_sources_footer, format_search_sources_footer
from .embedding.encoder import Encoder
from .ingestion.chunker import chunk_pages
from .ingestion.web_extractor import fetch_and_extract, url_to_source_name
from .perf import Timer
from .qa.generator_mlx import resolve_system_prompt, upstream_prompt_text
from .qa.token_usage import TokenUsage, add_usage, format_usage_plain
from .qa.retriever import RetrievedChunk
from .search import SearchResult, format_search_context, searxng_search
from .store.vector_store import (
    K_CHUNK_IDX, K_CHUNK_OVL, K_CHUNK_SIZE,
    K_DOC_DATE, K_DOC_DATE_Q,
    K_HASH, K_INGESTED, K_MODEL, K_PAGE, K_SOURCE,
    VectorStore,
)

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    import sys
    try:
        if sys.platform == "darwin":
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            return proc.returncode == 0
        elif sys.platform.startswith("linux"):
            for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
                try:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                    proc.communicate(text.encode("utf-8"))
                    if proc.returncode == 0:
                        return True
                except FileNotFoundError:
                    continue
        return False
    except Exception:
        return False


@dataclass
class QueryResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    embed_ms: float
    retrieve_ms: float
    generate_s: float
    turn_total_s: float
    usage: TokenUsage | None


@dataclass
class ChatTurn:
    user: str
    assistant: str


@dataclass
class LastResult:
    answer: str = ""
    sources: list[str] = field(default_factory=list)


_PROMPT_PREVIEW_PLACEHOLDER = (
    "After each answer, the system + user messages sent to the LLM "
    "appear here. Press Ctrl+Q to expand or collapse this panel.\n"
)

_HALF_PAGE_LINES = 15


class EmbdShell(App[None]):
    TITLE = "embd Interactive Shell"
    BINDINGS = [
        ("ctrl+q", "toggle_prompt_panel", "Prompt"),
        ("ctrl+shift+q", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear"),
    ]

    CSS = """
    #output {
        height: 1fr;
    }
    #output:focus {
        border: tall $accent;
    }
    #prompt_preview {
        height: 18;
        min-height: 8;
    }
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._cfg = cfg
        self._store = VectorStore(cfg.paths.db_dir, cfg.retrieval.collection_name)
        self._encoder = Encoder(cfg.embedding.model_name, cfg.embedding.device)
        self._generator = self._make_generator(cfg)
        self._busy = False
        self._history: list[ChatTurn] = []
        self._max_history_turns = 4
        self._last = LastResult()
        self._pending_g = False
        self._pending_y = False
        self._log = self._build_logger()
        self._session_prompt_tokens = 0
        self._session_completion_tokens = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static(
                f"-- INSERT --{self._base_status_tail()}",
                id="status_line",
            )
            yield TextArea(
                "",
                id="output",
                read_only=True,
                soft_wrap=True,
                show_line_numbers=False,
            )
            yield Collapsible(
                TextArea(
                    _PROMPT_PREVIEW_PLACEHOLDER,
                    id="prompt_preview",
                    read_only=True,
                    soft_wrap=True,
                    show_line_numbers=False,
                ),
                title="LLM upstream prompt (system + user)",
                collapsed=True,
                id="prompt_collapsible",
            )
            yield Input(
                placeholder="Ask a question or type /help — Esc to browse output",
                id="question_input",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self._llm_subtitle()
        self._append_output(
            "Interactive shell ready. Type /help for commands.\n"
            "Esc → browse output (cursor, shift+arrows to select, Ctrl+C to copy)\n"
            "i → back to input\n",
            scroll_to_start=True,
        )
        self._log.info("Shell started. backend=%s", self._cfg.llm.backend)

    def _llm_subtitle(self) -> str:
        """Short model / sampling summary for the window header (subtitle)."""
        b = self._cfg.llm.backend.strip().lower()
        if b == "mlx":
            m = self._cfg.llm.mlx_model
        elif b == "ollama":
            m = self._cfg.llm.ollama_model
        else:
            m = self._cfg.llm.claude_model
        if len(m) > 44:
            m = m[:20] + "…" + m[-20:]
        return (
            f"{b} · {m} · temp {self._cfg.llm.temperature} · "
            f"max_out {self._cfg.llm.max_tokens}"
        )

    def _base_status_tail(self) -> str:
        """Status line suffix: retrieval, embeddings, cumulative LLM tokens."""
        extra = ""
        if self._session_prompt_tokens or self._session_completion_tokens:
            st = self._session_prompt_tokens + self._session_completion_tokens
            extra = (
                f" | LLM Σ in/out/tot: "
                f"{self._session_prompt_tokens:,}/"
                f"{self._session_completion_tokens:,}/{st:,}"
            )
        return (
            f" | Top-K: {self._cfg.retrieval.top_k} | "
            f"Embeddings: {self._cfg.embedding.model_name}"
            f"{extra}"
        )

    # ------------------------------------------------------------------
    # Output helpers: append text and scroll so new content is read top-down
    # ------------------------------------------------------------------

    def _append_output(self, text: str, *, scroll_to_start: bool = True) -> None:
        """Append text to the output TextArea.

        By default the viewport jumps to the *start* of the new text so you
        read downward; set scroll_to_start=False to keep focus at the end
        (e.g. long pasted help).
        """
        out = self.query_one("#output", TextArea)
        start = out.document.end
        out.read_only = False
        out.insert(text, location=start)
        out.read_only = True
        if scroll_to_start:
            out.move_cursor(start)
        else:
            out.move_cursor(out.document.end)
        out.scroll_cursor_visible()

    def _set_prompt_preview(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> None:
        """Fill the collapsible panel with system + user text sent to the LLM."""
        sp = resolve_system_prompt(
            self._cfg.llm.system_prompt_preset,
            self._cfg.llm.system_prompt,
        )
        text = upstream_prompt_text(
            question,
            chunks,
            backend=self._cfg.llm.backend,
            system_prompt=sp,
        )
        ta = self.query_one("#prompt_preview", TextArea)
        ta.read_only = False
        ta.load_text(text)
        ta.read_only = True
        ta.move_cursor((0, 0))

    def action_toggle_prompt_panel(self) -> None:
        """Expand or collapse the upstream-prompt panel."""
        col = self.query_one("#prompt_collapsible", Collapsible)
        col.collapsed = not col.collapsed

    # ------------------------------------------------------------------
    # Vim-style navigation
    # ------------------------------------------------------------------

    def on_key(self, event) -> None:
        input_box = self.query_one("#question_input", Input)
        out = self.query_one("#output", TextArea)

        if event.key == "escape":
            if input_box.has_focus:
                out.focus()
                self._set_mode_indicator("NORMAL")
                event.prevent_default()
            return

        if not out.has_focus:
            return

        char = event.character or ""
        key = event.key

        if self._pending_g:
            self._pending_g = False
            if char == "g":
                out.move_cursor((0, 0))
                out.scroll_cursor_visible()
            self._set_mode_indicator("NORMAL")
            event.prevent_default()
            return

        if self._pending_y:
            self._pending_y = False
            if char == "y":
                self._copy_last_answer()
            elif char == "s":
                self._copy_last_sources()
            self._set_mode_indicator("NORMAL")
            event.prevent_default()
            return

        if char == "j":
            out.action_cursor_down()
            event.prevent_default()
        elif char == "k":
            out.action_cursor_up()
            event.prevent_default()
        elif char == "h":
            out.action_cursor_left()
            event.prevent_default()
        elif char == "l":
            out.action_cursor_right()
            event.prevent_default()
        elif char == "w":
            out.action_cursor_word_right()
            event.prevent_default()
        elif char == "b":
            out.action_cursor_word_left()
            event.prevent_default()
        elif char == "0":
            out.action_cursor_line_start()
            event.prevent_default()
        elif char == "$":
            out.action_cursor_line_end()
            event.prevent_default()
        elif char == "d" or key == "ctrl+d":
            for _ in range(_HALF_PAGE_LINES):
                out.action_cursor_down()
            out.scroll_cursor_visible()
            event.prevent_default()
        elif char == "u" or key == "ctrl+u":
            for _ in range(_HALF_PAGE_LINES):
                out.action_cursor_up()
            out.scroll_cursor_visible()
            event.prevent_default()
        elif char == "G":
            out.move_cursor(out.document.end)
            out.scroll_cursor_visible()
            event.prevent_default()
        elif char == "g":
            self._pending_g = True
            self._set_mode_indicator("NORMAL (g…)")
            event.prevent_default()
        elif char == "y":
            self._pending_y = True
            self._set_mode_indicator("NORMAL (y…)")
            event.prevent_default()
        elif char in ("i", "a", "o"):
            input_box.focus()
            self._set_mode_indicator("INSERT")
            event.prevent_default()
        elif char == "/":
            input_box.focus()
            input_box.value = "/search "
            input_box.action_end()
            self._set_mode_indicator("INSERT")
            event.prevent_default()
        elif char == ":":
            input_box.focus()
            input_box.value = "/"
            input_box.action_end()
            self._set_mode_indicator("INSERT")
            event.prevent_default()

    def _set_mode_indicator(self, mode: str) -> None:
        self.query_one("#status_line", Static).update(
            f"-- {mode} --{self._base_status_tail()}"
        )

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    @on(Input.Submitted, "#question_input")
    def on_submit(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        input_box = self.query_one("#question_input", Input)
        input_box.value = ""
        if self._busy or not question:
            return

        if question == "/quit":
            self.exit()
            return
        if question == "/clear":
            self.action_clear_log()
            return
        if question == "/help":
            self._write_help()
            return
        if question == "/reset":
            self._history.clear()
            self._append_output("Conversation history cleared.\n")
            return
        if question == "/copy":
            self._copy_last_answer()
            return
        if question == "/sources":
            self._copy_last_sources()
            return

        self._busy = True
        input_box.disabled = True

        if question.startswith("/search "):
            search_query = question[len("/search "):].strip()
            self._set_status("Working: searching the web...")
            self._run_search(search_query)
        elif question.startswith("/ingest "):
            url = question[len("/ingest "):].strip()
            self._set_status("Working: fetching URL...")
            self._run_ingest_url(url)
        else:
            self._set_status("Working: embedding query...")
            self._run_query(question)

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    @work(thread=True)
    def _run_query(self, question: str) -> None:
        try:
            start_s = time.perf_counter()
            prompt_question = self._build_prompt_question(question)
            self._log.info("Q: %s", question)

            with Timer("embed") as embed_timer:
                query_vec = self._encoder.encode_query(prompt_question)
            self.call_from_thread(self._set_status, "Working: retrieving chunks...")
            with Timer("retrieve") as retrieve_timer:
                hits = self._store.query(query_vec, top_k=self._cfg.retrieval.top_k)

            if not hits:
                self.call_from_thread(
                    self._append_output,
                    f"\n> {question}\n\n"
                    "No chunks found in the index. Run `embd ingest` first.\n",
                )
                self.call_from_thread(self._set_prompt_preview, prompt_question, [])
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

            self.call_from_thread(self._set_status, "Working: generating answer...")
            with Timer("generate") as gen_timer:
                answer, usage = self._generator.generate(prompt_question, chunks)

            turn_total_s = time.perf_counter() - start_s
            self._session_prompt_tokens, self._session_completion_tokens = add_usage(
                self._session_prompt_tokens,
                self._session_completion_tokens,
                usage,
            )

            result = QueryResult(
                question=question,
                answer=answer,
                chunks=chunks,
                embed_ms=embed_timer.elapsed * 1000,
                retrieve_ms=retrieve_timer.elapsed * 1000,
                generate_s=gen_timer.elapsed,
                turn_total_s=turn_total_s,
                usage=usage,
            )
            self._history.append(ChatTurn(user=question, assistant=answer))
            if len(self._history) > self._max_history_turns:
                self._history = self._history[-self._max_history_turns:]
            self._log.info(
                "A: len=%d chars, elapsed=%.2fs, hits=%d",
                len(answer), time.perf_counter() - start_s, len(chunks),
            )
            self.call_from_thread(self._set_prompt_preview, prompt_question, chunks)
            self.call_from_thread(self._render_result, result)
        except Exception as exc:
            self._log.exception("Query failed")
            self.call_from_thread(
                self._append_output,
                f"\n> {question}\n\n"
                f"Error: {type(exc).__name__}: {exc}\n"
                "Tip: verify `llm.mlx_model` exists on HF, "
                "or switch backend to `claude`/`ollama`.\n",
            )
        finally:
            self.call_from_thread(self._set_idle)

    @work(thread=True)
    def _run_search(self, search_query: str) -> None:
        try:
            start_s = time.perf_counter()
            self._log.info("Search: %s", search_query)
            results = searxng_search(
                search_query,
                base_url=self._cfg.search.searxng_url,
                max_results=self._cfg.search.max_results,
            )
            if not results:
                self.call_from_thread(
                    self._append_output,
                    f"\n> /search {search_query}\n\n"
                    "No web results found. Is your SearXNG instance running?\n",
                )
                return

            web_context = format_search_context(results)

            self.call_from_thread(self._set_status, "Working: embedding query...")
            with Timer("embed") as embed_timer:
                query_vec = self._encoder.encode_query(search_query)

            self.call_from_thread(self._set_status, "Working: retrieving local chunks...")
            with Timer("retrieve") as retrieve_timer:
                hits = self._store.query(query_vec, top_k=self._cfg.retrieval.top_k)

            local_chunks = [
                RetrievedChunk(
                    text=h["text"],
                    source_filename=h["metadata"]["source_filename"],
                    page_num=int(h["metadata"]["page_num"]),
                    chunk_index=int(h["metadata"]["chunk_index"]),
                    distance=h["distance"],
                )
                for h in hits
            ] if hits else []

            combined_question = self._build_prompt_question(search_query)
            if web_context:
                combined_question = (
                    f"{combined_question}\n\n"
                    f"Web search results:\n{web_context}"
                )

            self.call_from_thread(self._set_status, "Working: generating answer...")
            with Timer("generate") as gen_timer:
                answer, usage = self._generator.generate(
                    combined_question, local_chunks
                )

            turn_total_s = time.perf_counter() - start_s
            self._session_prompt_tokens, self._session_completion_tokens = add_usage(
                self._session_prompt_tokens,
                self._session_completion_tokens,
                usage,
            )

            self._history.append(ChatTurn(user=search_query, assistant=answer))
            if len(self._history) > self._max_history_turns:
                self._history = self._history[-self._max_history_turns:]

            self._log.info(
                "Search answer: len=%d chars, web_hits=%d, local_hits=%d",
                len(answer), len(results), len(local_chunks),
            )

            footer, source_lines = format_search_sources_footer(
                results, local_chunks, self._cfg.paths.documents_dir
            )
            self._last = LastResult(answer=answer, sources=source_lines)

            stats = format_usage_plain(
                usage,
                embed_ms=embed_timer.elapsed * 1000,
                retrieve_ms=retrieve_timer.elapsed * 1000,
                generate_s=gen_timer.elapsed,
                turn_total_s=turn_total_s,
                session_prompt=self._session_prompt_tokens,
                session_completion=self._session_completion_tokens,
            )
            block = (
                f"\n> /search {search_query}\n\n"
                + answer
                + footer
                + "\n\n"
                + stats
            )
            self.call_from_thread(
                self._set_prompt_preview, combined_question, local_chunks
            )
            self.call_from_thread(self._append_output, block)
        except Exception as exc:
            self._log.exception("Search failed")
            self.call_from_thread(
                self._append_output,
                f"\n> /search {search_query}\n\n"
                f"Search error: {type(exc).__name__}: {exc}\n",
            )
        finally:
            self.call_from_thread(self._set_idle)

    @work(thread=True)
    def _run_ingest_url(self, url: str) -> None:
        import hashlib
        from datetime import datetime, timezone

        try:
            self._log.info("Ingest URL: %s", url)
            result = fetch_and_extract(url)

            header = f"\n> /ingest {url}\n\n"
            if not result.ok:
                self.call_from_thread(
                    self._append_output,
                    header + result.summary() + "\n",
                )
                return

            source_name = url_to_source_name(url)
            self._store.delete_file(source_name)

            chunks = chunk_pages(
                result.pages,
                source_filename=source_name,
                chunk_size=self._cfg.ingestion.chunk_size,
                chunk_overlap=self._cfg.ingestion.chunk_overlap,
            )
            texts = [c.text for c in chunks]

            self.call_from_thread(
                self._set_status,
                f"Working: embedding {len(chunks)} chunks...",
            )
            embeddings = self._encoder.encode(texts)

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
                    K_MODEL:      self._encoder.model_version,
                    K_CHUNK_SIZE: self._cfg.ingestion.chunk_size,
                    K_CHUNK_OVL:  self._cfg.ingestion.chunk_overlap,
                    K_DOC_DATE:   (doc_date.date or "") if doc_date else "",
                    K_DOC_DATE_Q: (doc_date.quality) if doc_date else "none",
                }
                for c in chunks
            ]
            self._store.upsert_chunks(
                chunk_ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                texts=texts,
                metadatas=metadatas,
            )
            self._log.info("Ingested %d chunks from %s", len(chunks), url)
            self.call_from_thread(
                self._append_output,
                header
                + result.summary()
                + "\n\n"
                + f"Ingested {len(chunks)} chunks (source: {source_name})\n",
            )
        except Exception as exc:
            self._log.exception("Ingest URL failed")
            self.call_from_thread(
                self._append_output,
                f"\n> /ingest {url}\n\n"
                f"Ingest error: {type(exc).__name__}: {exc}\n",
            )
        finally:
            self.call_from_thread(self._set_idle)

    # ------------------------------------------------------------------
    # Clipboard
    # ------------------------------------------------------------------

    def _copy_last_answer(self) -> None:
        if not self._last.answer:
            self._append_output("Nothing to copy yet — ask a question first.\n")
            return
        if _copy_to_clipboard(self._last.answer):
            self._append_output("Answer copied to clipboard.\n")
        else:
            self._append_output("Clipboard not available.\n")

    def _copy_last_sources(self) -> None:
        if not self._last.sources:
            self._append_output("No sources yet — ask a question first.\n")
            return
        text = "\n".join(self._last.sources)
        if _copy_to_clipboard(text):
            self._append_output("Sources copied to clipboard.\n")
        else:
            self._append_output("Clipboard not available.\n")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _set_idle(self) -> None:
        self._busy = False
        input_box = self.query_one("#question_input", Input)
        input_box.disabled = False
        input_box.focus()
        self._set_mode_indicator("INSERT")

    def _set_status(self, text: str) -> None:
        self.query_one("#status_line", Static).update(f"-- WORKING -- | {text}")

    def _render_result(self, result: QueryResult) -> None:
        footer, source_lines = format_local_sources_footer(
            result.chunks, self._cfg.paths.documents_dir
        )
        self._last = LastResult(answer=result.answer, sources=source_lines)

        stats = format_usage_plain(
            result.usage,
            embed_ms=result.embed_ms,
            retrieve_ms=result.retrieve_ms,
            generate_s=result.generate_s,
            turn_total_s=result.turn_total_s,
            session_prompt=self._session_prompt_tokens,
            session_completion=self._session_completion_tokens,
        )
        block = (
            f"\n> {result.question}\n\n"
            + result.answer
            + footer
            + "\n\n"
            + stats
        )
        self._append_output(block)

    def action_clear_log(self) -> None:
        out = self.query_one("#output", TextArea)
        out.read_only = False
        out.clear()
        out.read_only = True
        ta = self.query_one("#prompt_preview", TextArea)
        ta.read_only = False
        ta.load_text(_PROMPT_PREVIEW_PLACEHOLDER)
        ta.read_only = True

    def _write_help(self) -> None:
        self._append_output(
            "\n"
            "Commands (type in input box):\n"
            "  <question>         ask against local documents\n"
            "  /search <query>    web search via SearXNG + local docs\n"
            "  /ingest <url>      fetch & index a web page\n"
            "  /copy              copy last answer to clipboard\n"
            "  /sources           copy last source list to clipboard\n"
            "  /help              show this help\n"
            "  /clear             clear output\n"
            "  /reset             clear conversation memory\n"
            "  /quit  or Ctrl+Shift+Q   exit shell\n"
            "  Ctrl+Q                 toggle LLM upstream prompt panel\n"
            "\n"
            "Output navigation (press Esc to focus output):\n"
            "  Arrow keys         move cursor\n"
            "  Shift+arrows       select text\n"
            "  Ctrl+C             copy selection to clipboard\n"
            "  j/k/h/l            vim cursor movement\n"
            "  w/b                word forward/back\n"
            "  0 / $              line start/end\n"
            "  d / u              half-page down/up\n"
            "  gg                 go to top\n"
            "  G                  go to bottom\n"
            "  yy                 yank answer to clipboard\n"
            "  ys                 yank sources to clipboard\n"
            "  i / a / o          back to input box\n"
            "  /                  start a /search\n"
            "  :                  start a /command\n"
            "\n"
            "Logs: ./embd_shell.log\n"
            "\n"
            "After each LLM answer, a plain-text block shows timing, tokens in/out/total "
            "for that question, and cumulative LLM tokens since the shell started. "
            "Model and sampling (temp, max output) are in the window subtitle.\n"
            "\n"
            "Live index: if you run `embd ingest` or `embd serve` (with its file watcher) "
            "in another terminal, new/updated chunks are visible to the next query here "
            "without restarting the shell.\n",
            scroll_to_start=False,
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt_question(self, question: str) -> str:
        if not self._history:
            return question
        history_lines: list[str] = []
        for i, turn in enumerate(self._history[-self._max_history_turns:], start=1):
            history_lines.append(f"Turn {i} User: {turn.user}")
            history_lines.append(f"Turn {i} Assistant: {turn.assistant}")
        return (
            "Conversation so far:\n"
            + "\n".join(history_lines)
            + f"\n\nCurrent user question: {question}"
        )

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("embd.shell")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        log_path = Path.cwd() / "embd_shell.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        return logger

    @staticmethod
    def _make_generator(cfg: Config):
        from .qa.generator_mlx import resolve_system_prompt

        backend = cfg.llm.backend.strip().lower()
        system_prompt = resolve_system_prompt(
            cfg.llm.system_prompt_preset,
            cfg.llm.system_prompt,
        )
        if backend == "mlx":
            from .qa.generator_mlx import MLXGenerator
            return MLXGenerator(
                model_path=cfg.llm.mlx_model,
                temperature=cfg.llm.temperature,
                max_tokens=cfg.llm.max_tokens,
                system_prompt=system_prompt,
            )
        if backend == "ollama":
            from .qa.generator_ollama import OllamaGenerator
            return OllamaGenerator(
                model=cfg.llm.ollama_model,
                host=cfg.llm.ollama_host,
                temperature=cfg.llm.temperature,
                num_predict=cfg.llm.max_tokens,
                system_prompt=system_prompt,
            )
        if backend == "claude":
            from .qa.generator_claude import ClaudeGenerator
            return ClaudeGenerator(
                model=cfg.llm.claude_model,
                api_key=cfg.llm.claude_api_key,
                temperature=cfg.llm.temperature,
                max_tokens=cfg.llm.max_tokens,
                system_prompt=system_prompt,
            )
        raise ValueError(
            f"Unknown LLM backend: '{cfg.llm.backend}'. "
            "Set backend to 'mlx', 'ollama', or 'claude' in config.toml."
        )
