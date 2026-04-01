"""Word document (.docx) extraction via python-docx.

Requires the optional ``python-docx`` package:
    pip install 'embd[docx]'

Extracts all paragraph text and, when OCR is available, also extracts
embedded images from the DOCX ZIP archive and OCR's them.  All content
is returned as a single PageText (page_num=1); the downstream chunker
handles splitting.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from ..config import IngestionConfig
from .extractor import PageText
from .ocr import is_ocr_available, ocr_images, preload_surya_for_ingestion

logger = logging.getLogger(__name__)

_MIN_IMAGE_BYTES = 2_000


def _extract_docx_images(path: Path, *, max_images: int) -> list[bytes]:
    """Pull image files from the word/media/ folder inside the DOCX ZIP.

    Largest images first; ``max_images`` limits how many are returned (0 = no limit).
    """
    images: list[bytes] = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for entry in zf.filelist:
                if not entry.filename.startswith("word/media/"):
                    continue
                lower = entry.filename.lower()
                if not any(lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")):
                    continue
                data = zf.read(entry.filename)
                if len(data) >= _MIN_IMAGE_BYTES:
                    images.append(data)
    except Exception as exc:
        logger.debug("Could not read images from '%s': %s", path.name, exc)
    images.sort(key=len, reverse=True)
    if max_images > 0 and len(images) > max_images:
        images = images[:max_images]
    return images


def extract_docx_pages(path: Path, ingestion: IngestionConfig) -> list[PageText]:
    """Extract text (and optionally OCR'd image text) from a .docx file."""
    try:
        from docx import Document  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "python-docx is required for .docx files. "
            "Install with: pip install 'embd[docx]'"
        )
        return []

    try:
        doc = Document(str(path))
    except Exception as exc:
        logger.warning("Cannot read '%s': %s — skipping file.", path.name, exc)
        return []

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs).strip()

    mode = (ingestion.ocr_embedded_images or "when_no_text").strip().lower()
    if mode not in ("always", "when_no_text", "never"):
        mode = "when_no_text"
    if mode != "never" and is_ocr_available(ocr_backend=ingestion.ocr_backend):
        run_img_ocr = mode == "always" or (mode == "when_no_text" and not text)
        if run_img_ocr:
            max_img = max(0, int(ingestion.ocr_max_images_per_page or 0))
            img_bytes = _extract_docx_images(path, max_images=max_img)
            if img_bytes:
                preload_surya_for_ingestion(ingestion.ocr_backend)
                logger.info("OCR'ing %d image(s) from '%s'", len(img_bytes), path.name)
                ocr_text = ocr_images(
                    img_bytes,
                    lang=ingestion.ocr_tesseract_lang,
                    ocr_backend=ingestion.ocr_backend,
                )
                if ocr_text:
                    text = (text + "\n\n" + ocr_text).strip() if text else ocr_text

    if not text:
        logger.warning("'%s' yielded no extractable text.", path.name)
        return []

    return [PageText(page_num=1, text=text)]
