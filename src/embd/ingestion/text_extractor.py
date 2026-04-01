"""Plain-text and markdown extraction.

Reads the entire file as UTF-8 into a single PageText (page_num=1).
The downstream chunker handles splitting into appropriately sized pieces.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ..config import IngestionConfig
from .extractor import PageText

logger = logging.getLogger(__name__)


def extract_text_pages(path: Path, _ingestion: IngestionConfig) -> list[PageText]:
    """Read a plain-text or markdown file and return it as one page."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        logger.warning("Cannot read '%s': %s — skipping file.", path.name, exc)
        return []

    if not text:
        logger.warning("'%s' is empty.", path.name)
        return []

    return [PageText(page_num=1, text=text)]
