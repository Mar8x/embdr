"""Extractor registry: maps config type names to file extensions and extract functions.

The scanner, ingest module, and file watcher all use this registry so there is
a single source of truth for which file types are supported and how to extract them.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from ..config import IngestionConfig
from .extractor import PageText

logger = logging.getLogger(__name__)

ExtractFn = Callable[[Path, IngestionConfig], list[PageText]]

_REGISTRY: dict[str, tuple[set[str], ExtractFn]] = {}
_INITIALIZED = False


def _ensure_registry() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    from .docx_extractor import extract_docx_pages
    from .epub_extractor import extract_epub_pages
    from .extractor import extract_pages as extract_pdf_pages
    from .text_extractor import extract_text_pages
    from .xlsx_extractor import extract_xlsx_pages

    _REGISTRY.update({
        "pdf":  ({".pdf"}, extract_pdf_pages),
        "epub": ({".epub"}, extract_epub_pages),
        "txt":  ({".txt", ".text", ".rst", ".log"}, extract_text_pages),
        "md":   ({".md", ".markdown"}, extract_text_pages),
        "docx": ({".docx"}, extract_docx_pages),
        "xlsx": ({".xlsx", ".xls"}, extract_xlsx_pages),
    })


def get_supported_extensions(enabled_types: list[str]) -> set[str]:
    """Return the union of file extensions for all *enabled_types*."""
    _ensure_registry()
    exts: set[str] = set()
    for t in enabled_types:
        entry = _REGISTRY.get(t.strip().lower())
        if entry is not None:
            exts |= entry[0]
        else:
            logger.warning("Unknown ingestion type '%s' in config — ignored.", t)
    return exts


def extract_file(path: Path, ingestion: IngestionConfig) -> list[PageText]:
    """Dispatch extraction based on file suffix and enabled types.

    Returns an empty list when the suffix is not covered by any enabled type
    or when the extractor finds no text.
    """
    _ensure_registry()
    suffix = path.suffix.lower()
    for t in ingestion.enabled_types:
        entry = _REGISTRY.get(t.strip().lower())
        if entry is not None and suffix in entry[0]:
            return entry[1](path, ingestion)
    logger.debug("No enabled extractor for '%s' (suffix %s)", path.name, suffix)
    return []


def list_types() -> dict[str, set[str]]:
    """Return {type_name: {extensions}} for documentation / help."""
    _ensure_registry()
    return {name: exts for name, (exts, _) in _REGISTRY.items()}
