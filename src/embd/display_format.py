"""
Shared formatting for answer output: footnote-style source sections.

Keeps CLI and shell visually consistent.
"""
from __future__ import annotations

from pathlib import Path

from .qa.retriever import RetrievedChunk
from .search import SearchResult

# Narrow rule — reads like a footnote separator in monospace TUIs
_RULE = "─" * 58


def format_local_sources_footer(
    chunks: list[RetrievedChunk],
    documents_dir: Path,
) -> tuple[str, list[str]]:
    """Return (display block, lines for /sources clipboard).

    Deduplicates by (filename, page). Numbered like footnotes [1], [2], …
    """
    seen: set[str] = set()
    entries: list[str] = []
    clipboard: list[str] = []
    n = 0
    doc_resolved = documents_dir.resolve()
    for c in chunks:
        key = f"{c.source_filename}:p{c.page_num}"
        if key in seen:
            continue
        seen.add(key)
        n += 1
        path = doc_resolved / c.source_filename
        date_str = ""
        if c.document_date:
            date_str = f"  ·  date {c.document_date[:10]} ({c.document_date_quality})"
        entries.append(
            f"  [{n}]  {c.source_filename}\n"
            f"       p.{c.page_num}  ·  {path}\n"
            f"       relevance (distance) {c.distance:.4f}{date_str}"
        )
        clipboard.append(f"[{n}] {c.source_filename} p.{c.page_num} — {path}")
    if not entries:
        return "", []
    body = "\n\n".join(entries)
    block = (
        f"\n\n{_RULE}\n"
        f"  Sources\n"
        f"{_RULE}\n\n"
        f"{body}\n"
    )
    return block, clipboard


def format_search_sources_footer(
    web_results: list[SearchResult],
    local_chunks: list[RetrievedChunk],
    documents_dir: Path,
) -> tuple[str, list[str]]:
    """Footnote-style block for /search: web URLs then local doc paths."""
    clipboard: list[str] = []
    sections: list[str] = []
    n = 0

    if web_results:
        web_entries: list[str] = []
        for r in web_results:
            n += 1
            title = r.title.strip() or "(no title)"
            web_entries.append(
                f"  [{n}]  {title}\n"
                f"       {r.url}"
            )
            clipboard.append(f"[{n}] {r.url}")
        sections.append("  Web\n\n" + "\n\n".join(web_entries))

    if local_chunks:
        seen: set[str] = set()
        local_entries: list[str] = []
        doc_resolved = documents_dir.resolve()
        for c in local_chunks:
            key = f"{c.source_filename}:p{c.page_num}"
            if key in seen:
                continue
            seen.add(key)
            n += 1
            path = doc_resolved / c.source_filename
            date_str = ""
            if c.document_date:
                date_str = f"  ·  date {c.document_date[:10]} ({c.document_date_quality})"
            local_entries.append(
                f"  [{n}]  {c.source_filename}\n"
                f"       p.{c.page_num}  ·  {path}\n"
                f"       relevance (distance) {c.distance:.4f}{date_str}"
            )
            clipboard.append(f"[{n}] {c.source_filename} p.{c.page_num} — {path}")
        sections.append(
            "  Local documents\n\n" + "\n\n".join(local_entries)
        )

    if not sections:
        return "", []

    inner = "\n\n".join(sections)
    block = (
        f"\n\n{_RULE}\n"
        f"  Sources\n"
        f"{_RULE}\n\n"
        f"{inner}\n"
    )
    return block, clipboard
