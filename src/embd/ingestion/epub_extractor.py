"""
EPUB text extraction using ebooklib + BeautifulSoup.

Extracts text chapter-by-chapter, treating each XHTML document item
as a "page" for downstream metadata. Chapters with no extractable
text are silently skipped.
"""
from __future__ import annotations

import logging
from pathlib import Path

from bs4 import BeautifulSoup

from ..config import IngestionConfig
from .extractor import PageText

logger = logging.getLogger(__name__)


def extract_epub_pages(epub_path: Path, _ingestion: IngestionConfig) -> list[PageText]:
    """Extract text from all chapters of an EPUB, returning only non-empty ones.

    Each XHTML document item in the EPUB spine is treated as one page.
    Page numbers are 1-indexed sequential chapter numbers.
    """
    import ebooklib  # type: ignore[import-untyped]
    from ebooklib import epub  # type: ignore[import-untyped]

    pages: list[PageText] = []

    try:
        book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})
    except Exception as exc:
        logger.warning(
            "Cannot read '%s': %s — skipping file.",
            epub_path.name, exc,
        )
        return []

    chapter_num = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        chapter_num += 1
        try:
            html = item.get_content().decode("utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            text = "\n".join(line for line in text.splitlines() if line.strip())
        except Exception as exc:
            logger.debug(
                "Chapter %d of '%s': extraction failed (%s), skipping",
                chapter_num, epub_path.name, exc,
            )
            continue

        if not text:
            logger.debug(
                "Chapter %d of '%s': no extractable text",
                chapter_num, epub_path.name,
            )
            continue

        pages.append(PageText(page_num=chapter_num, text=text))

    if not pages:
        logger.warning(
            "'%s' yielded no extractable text.",
            epub_path.name,
        )

    return pages
