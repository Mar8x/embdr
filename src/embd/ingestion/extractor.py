"""
PDF text extraction using PyMuPDF (fitz).

PyMuPDF is 3-7x faster than pdfplumber for text extraction. It uses
MuPDF's C library under the hood and handles most PDF variants well.

Extracts text page-by-page, preserving page numbers for downstream
metadata. Optional embedded-image OCR follows ``IngestionConfig``
(``ocr_embedded_images``, ``ocr_backend``).  Pages with no extractable
text at all are skipped. If an entire PDF yields no text, a warning
is logged and the caller should skip the file.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from ..config import IngestionConfig
from .ocr import is_ocr_available, ocr_images, preload_surya_for_ingestion

logger = logging.getLogger(__name__)

_SLOW_PAGE_THRESHOLD = 5.0
_MIN_IMAGE_BYTES = 2_000
_warned_always_ocr = False


@dataclass
class PageText:
    page_num: int   # 1-indexed, matching standard PDF page numbering
    text: str


def _extract_page_images(page: fitz.Page, *, max_images: int) -> list[bytes]:
    """Return raw image bytes for every non-trivial image on *page*.

    Images are sorted largest-first so OCR budget (``max_images``) favours figures over icons.
    """
    images: list[bytes] = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base_image = page.parent.extract_image(xref)
            if base_image and len(base_image["image"]) >= _MIN_IMAGE_BYTES:
                images.append(base_image["image"])
        except Exception:
            continue
    images.sort(key=len, reverse=True)
    if max_images > 0 and len(images) > max_images:
        images = images[:max_images]
    return images


def extract_pages(pdf_path: Path, ingestion: IngestionConfig) -> list[PageText]:
    """Extract text from all pages of a PDF, returning only non-empty pages.

    Uses PyMuPDF's get_text("text") which is significantly faster than
    pdfplumber's layout analysis. The "text" mode preserves reading order
    and handles multi-column layouts reasonably well.

    Embedded-image OCR follows ``ingestion.ocr_embedded_images`` (see config).

    Returns an empty list if the file cannot be opened or contains no
    extractable text. The caller skips the file in both cases.
    """
    pages: list[PageText] = []
    mode = (ingestion.ocr_embedded_images or "when_no_text").strip().lower()
    if mode not in ("always", "when_no_text", "never"):
        logger.warning(
            "Unknown ingestion.ocr_embedded_images=%r — using when_no_text",
            ingestion.ocr_embedded_images,
        )
        mode = "when_no_text"
    if mode == "never":
        do_ocr = False
    else:
        do_ocr = is_ocr_available(ocr_backend=ingestion.ocr_backend)

    global _warned_always_ocr
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        logger.warning(
            "Cannot read '%s': %s — skipping file. "
            "The file may be corrupt, password-protected, or not a valid PDF.",
            pdf_path.name, exc,
        )
        return []

    try:
        total_pages = len(doc)
        max_img = max(0, int(ingestion.ocr_max_images_per_page or 0))
        ocr_note = "off" if not do_ocr else f"on ({mode}, backend={ingestion.ocr_backend})"
        if do_ocr and max_img > 0:
            ocr_note += f", max {max_img} img/page"
        logger.info(
            "Extracting text from '%s' (%d pages, image OCR %s) …",
            pdf_path.name, total_pages, ocr_note,
        )
        if do_ocr and mode == "always" and not _warned_always_ocr:
            _warned_always_ocr = True
            logger.warning(
                "ocr_embedded_images=always runs OCR on every embedded image on every page — "
                "often 10–100× slower than when_no_text; use when_no_text unless you need "
                "figures on pages that already have a text layer."
            )
            # Preload eagerly for "always" mode since every page will need it
            preload_surya_for_ingestion(ingestion.ocr_backend)
        ocr_preloaded = mode == "always"
        t0 = time.perf_counter()

        for i, page in enumerate(doc, start=1):
            page_start = time.perf_counter()
            try:
                text = page.get_text("text").strip()
            except Exception as page_exc:
                logger.debug(
                    "Page %d of '%s': extraction failed (%s), skipping page",
                    i, pdf_path.name, page_exc,
                )
                text = ""

            if do_ocr:
                run_img_ocr = mode == "always" or (mode == "when_no_text" and not text)
                if run_img_ocr:
                    # Lazy-init: only load Surya when a page actually needs OCR
                    if not ocr_preloaded:
                        preload_surya_for_ingestion(ingestion.ocr_backend)
                        ocr_preloaded = True
                    img_bytes = _extract_page_images(page, max_images=max_img)
                    if img_bytes:
                        ocr_text = ocr_images(
                            img_bytes,
                            lang=ingestion.ocr_tesseract_lang,
                            ocr_backend=ingestion.ocr_backend,
                        )
                        if ocr_text:
                            text = (text + "\n\n" + ocr_text).strip() if text else ocr_text

            page_elapsed = time.perf_counter() - page_start
            if page_elapsed > _SLOW_PAGE_THRESHOLD:
                logger.warning(
                    "Page %d of '%s' took %.1fs (complex layout or large images?)",
                    i, pdf_path.name, page_elapsed,
                )

            if not text:
                continue
            pages.append(PageText(page_num=i, text=text))

        elapsed = time.perf_counter() - t0
        logger.info(
            "Extracted %d pages from '%s' in %.1fs",
            len(pages), pdf_path.name, elapsed,
        )
    finally:
        doc.close()

    if not pages:
        logger.warning(
            "'%s' yielded no extractable text. "
            "This PDF may be scanned or image-based. "
            "Install OCR (see README) or set ocr_embedded_images / ocr_backend in config, "
            "or pre-process with ocrmypdf.",
            pdf_path.name,
        )

    return pages
