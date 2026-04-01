"""
Sentence-aware text chunker.

Splits extracted page text into chunks that respect sentence boundaries,
then assigns each chunk a stable, position-based ID that enables safe
ChromaDB upsert (idempotent for unchanged files) and efficient per-file
deletion.

Compared to a naive character-level sliding window, sentence-aware
chunking avoids cutting mid-sentence, which significantly improves
retrieval quality because each chunk is a coherent passage.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .extractor import PageText


@dataclass
class Chunk:
    chunk_id: str          # stable 32-hex-char ID (position-based SHA-256)
    text: str
    source_filename: str
    page_num: int          # page where this chunk begins
    chunk_index: int       # 0-based sequential index within the document


def _make_chunk_id(filename: str, page_num: int, chunk_index: int) -> str:
    """Generate a stable, globally unique chunk ID.

    Formula: sha256(f"{filename}::{page_num}::{chunk_index}").hexdigest()[:32]

    Why position-based rather than content-based:
    - Content-based IDs (hash of text) would prevent per-file deletion via
      metadata filter alone -- we'd need to scan all IDs to find a file's chunks.
    - Position-based IDs are deterministic for unchanged files, so re-ingesting
      an unchanged document is a ChromaDB upsert no-op (same IDs, same vectors).
    - 32 hex chars = 128 bits of entropy; birthday collision probability is
      negligible for any realistic document collection.
    """
    key = f"{filename}::{page_num}::{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


_SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?])\s+'       # split after sentence-ending punctuation + whitespace
    r'|(?<=\n)\n+'         # or at paragraph boundaries (double newline)
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving non-empty segments."""
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip()]


def _split_sentences_with_offsets(text: str) -> tuple[list[str], list[int]]:
    """Split text into sentences and return their character offsets.

    Returns (sentences, offsets) where offsets[i] is the position of
    sentences[i] in the original text. Used for page-number mapping
    without a full-text search.
    """
    sentences: list[str] = []
    offsets: list[int] = []
    for m in _SENTENCE_SPLIT.finditer(text):
        pass  # we need the split positions

    # re.split doesn't give positions, so use finditer on the gaps instead.
    parts: list[tuple[int, int]] = []
    prev = 0
    for m in _SENTENCE_SPLIT.finditer(text):
        parts.append((prev, m.start()))
        prev = m.end()
    parts.append((prev, len(text)))

    for start, end in parts:
        s = text[start:end].strip()
        if s:
            real_start = start + (text[start:end].index(s[0]) if s else 0)
            sentences.append(s)
            offsets.append(real_start)

    return sentences, offsets


def chunk_pages(
    pages: list[PageText],
    source_filename: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split pages into sentence-aware chunks.

    Strategy:
    1. Join all pages into flat text with page-offset tracking.
    2. Split the full text into sentences, recording each sentence's
       character offset so we can map back to the correct page.
    3. Greedily pack sentences into chunks up to chunk_size characters.
    4. For overlap, the last few sentences of the previous chunk are
       prepended to the next chunk (up to chunk_overlap characters),
       but always advancing by at least one new sentence to guarantee
       forward progress.

    This ensures no sentence is ever split across chunks.
    """
    segments: list[tuple[int, int, int]] = []
    full_text = ""
    for page in pages:
        start = len(full_text)
        full_text += page.text + "\n\n"
        segments.append((start, len(full_text), page.page_num))

    sentences, sent_offsets = _split_sentences_with_offsets(full_text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    chunk_index = 0
    i = 0

    while i < len(sentences):
        start_i = i
        current_sentences: list[str] = []
        current_len = 0

        while i < len(sentences):
            sent = sentences[i]
            added_len = len(sent) + (1 if current_sentences else 0)
            if current_sentences and current_len + added_len > chunk_size:
                break
            current_sentences.append(sent)
            current_len += added_len
            i += 1

        if not current_sentences:
            break

        chunk_text = " ".join(current_sentences)
        page_num = _page_for_offset(segments, sent_offsets[start_i])

        chunks.append(Chunk(
            chunk_id=_make_chunk_id(source_filename, page_num, chunk_index),
            text=chunk_text,
            source_filename=source_filename,
            page_num=page_num,
            chunk_index=chunk_index,
        ))
        chunk_index += 1

        if i >= len(sentences):
            break

        # Overlap: rewind so trailing sentences from the current chunk
        # are shared with the next, but never rewind past start_i + 1
        # (must advance by at least one sentence to avoid infinite loops).
        overlap_len = 0
        rewind = 0
        for j in range(len(current_sentences) - 1, -1, -1):
            slen = len(current_sentences[j]) + 1
            if overlap_len + slen > chunk_overlap:
                break
            overlap_len += slen
            rewind += 1
        max_rewind = i - start_i - 1
        rewind = min(rewind, max_rewind)
        if rewind > 0:
            i -= rewind

    return chunks


def _page_for_offset(segments: list[tuple[int, int, int]], offset: int) -> int:
    """Return the page number that contains the given character offset."""
    for seg_start, seg_end, page_num in segments:
        if seg_start <= offset < seg_end:
            return page_num
    return segments[-1][2]
