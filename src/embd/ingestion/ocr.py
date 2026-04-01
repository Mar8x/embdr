"""Shared OCR helper with configurable backend selection.

Backends:

- **Surya** — high-quality document OCR (PyTorch); uses MPS (Apple Silicon) or CUDA when
  available. Slower than Tesseract but much better accuracy on typical PDFs.
  Requires ``transformers>=4.56,<5`` (Surya breaks on transformers 5.x today).
- **Tesseract** — CPU; fast baseline when no GPU or when forced via config.

Configure via ``[ingestion].ocr_backend`` in config.toml (``auto`` | ``surya`` | ``tesseract``).
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Literal

logger = logging.getLogger(__name__)

_MIN_IMAGE_PIXELS = 40 * 40
_SURYA_MAX_WIDTH = 2048  # Surya docs: very wide images can hurt quality / memory

Backend = Literal["surya", "tesseract", "none"]

_backend_cache: dict[str, Backend] = {}
# Lazy Surya stack: FoundationPredictor + RecognitionPredictor + DetectionPredictor
_surya_bundle: tuple[object, object, object] | None = None
_surya_init_lock = threading.Lock()
# Set True if Surya model init failed; skip Surya for the rest of the process.
_surya_unavailable: bool = False


def _try_tesseract() -> Backend:
    try:
        import pytesseract
        from PIL import Image  # noqa: F401

        pytesseract.get_tesseract_version()
        return "tesseract"
    except Exception:
        return "none"


def _try_surya_gpu() -> Backend:
    """Surya is only selected when MPS or CUDA is available (quality path; CPU is too slow)."""
    if _surya_unavailable:
        return "none"
    try:
        import torch

        if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
            return "none"
        import surya  # noqa: F401
    except ImportError:
        return "none"
    device = "MPS" if torch.backends.mps.is_available() else "CUDA"
    logger.info("OCR backend: Surya (PyTorch on %s)", device)
    return "surya"


def _detect_backend(preference: str) -> Backend:
    """Resolve backend from ``[ingestion].ocr_backend``."""
    raw = (preference or "auto").strip().lower()
    if raw == "easyocr":
        logger.warning(
            "ocr_backend=easyocr is no longer supported — use surya or auto (Surya on GPU)"
        )
        raw = "auto"

    pref = raw
    if pref not in ("auto", "surya", "tesseract"):
        logger.warning("Unknown ingestion.ocr_backend=%r — using auto", preference)
        pref = "auto"

    if pref == "tesseract":
        b = _try_tesseract()
        if b != "none":
            logger.info("OCR backend: Tesseract (CPU) — forced by ocr_backend setting")
        return b

    if pref == "surya":
        b = _try_surya_gpu()
        if b != "none":
            return b
        if _surya_unavailable:
            logger.info(
                "ocr_backend=surya but models failed to load — falling back to Tesseract"
            )
        else:
            logger.info("ocr_backend=surya but no MPS/CUDA — falling back to Tesseract")
        return _try_tesseract()

    # auto: Surya on GPU, else Tesseract
    b = _try_surya_gpu()
    if b != "none":
        return b
    b = _try_tesseract()
    if b != "none":
        if _surya_unavailable:
            logger.info(
                "OCR backend: Tesseract (CPU) — Surya disabled (load failed; "
                "try: pip install 'transformers>=4.56.1,<5')"
            )
        else:
            logger.info("OCR backend: Tesseract (CPU) — no GPU for Surya")
        return b
    logger.debug("OCR not available (install embd[ocr] or embd[ocr-cpu] + Tesseract)")
    return "none"


def _get_backend(preference: str = "auto") -> Backend:
    if preference not in _backend_cache:
        _backend_cache[preference] = _detect_backend(preference)
    return _backend_cache[preference]


def _invalidate_backend_cache() -> None:
    """After Surya fails to load, re-resolve auto/surya to Tesseract."""
    _backend_cache.clear()


def _get_surya_bundle() -> tuple[object, object, object] | None:
    """Load Surya once per process, or return None if loading failed.

    Always take the lock before reading ``_surya_bundle`` (no unlocked fast path):
    an unlocked check-then-lock pattern lets many threads all pass ``is None`` and
    each log + load models (same wall-clock second, apparent hang).

    After the first load, the lock is uncontended and cheap.
    """
    global _surya_bundle, _surya_unavailable
    with _surya_init_lock:
        if _surya_unavailable:
            return None
        if _surya_bundle is not None:
            return _surya_bundle
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        logger.info(
            "Initialising Surya OCR (first call — model download may take several minutes) …"
        )
        try:
            foundation = FoundationPredictor()
            recognition = RecognitionPredictor(foundation)
            detection = DetectionPredictor()
        except Exception:
            _surya_unavailable = True
            _invalidate_backend_cache()
            logger.exception(
                "Surya OCR failed to initialize (often incompatible ``transformers``: "
                "use ``pip install 'transformers>=4.56.1,<5'`` with ``embd[ocr]``). "
                "Falling back to Tesseract for embedded-image OCR in this process."
            )
            return None

        _surya_bundle = (foundation, recognition, detection)
        logger.info("Surya OCR predictors ready.")
        return _surya_bundle


def preload_surya_for_ingestion(ocr_backend: str) -> None:
    """If the resolved backend is Surya, load models once before page/image loops.

    Call from PDF/DOCX extractors when embedded-image OCR is enabled so the first
    heavy load does not coincide with many ``ocr_image`` calls.
    """
    if _get_backend(ocr_backend) == "surya":
        _get_surya_bundle()


# ── public API ───────────────────────────────────────────────────────


def is_ocr_available(*, ocr_backend: str = "auto") -> bool:
    """Return True if any OCR backend is usable for the given preference."""
    return _get_backend(ocr_backend) != "none"


def ocr_image(image_bytes: bytes, *, lang: str = "eng", ocr_backend: str = "auto") -> str:
    """Run OCR on raw image bytes and return extracted text."""
    backend = _get_backend(ocr_backend)
    if backend == "none":
        return ""

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if backend == "surya":
            bundle = _get_surya_bundle()
            if bundle is not None:
                return _ocr_surya(img, bundle)
            backend = _get_backend(ocr_backend)

        return _ocr_tesseract(img, lang=lang)
    except Exception as exc:
        logger.debug("OCR failed on image: %s", exc)
        return ""


def ocr_images(
    image_list: list[bytes],
    *,
    lang: str = "eng",
    ocr_backend: str = "auto",
) -> str:
    """OCR a list of raw image byte-strings and return the combined text."""
    if not is_ocr_available(ocr_backend=ocr_backend) or not image_list:
        return ""

    if _get_backend(ocr_backend) == "surya":
        _get_surya_bundle()

    from PIL import Image

    parts: list[str] = []
    for raw in image_list:
        try:
            img = Image.open(io.BytesIO(raw))
            w, h = img.size
            if w * h < _MIN_IMAGE_PIXELS:
                continue
        except Exception:
            continue
        text = ocr_image(raw, lang=lang, ocr_backend=ocr_backend)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


# ── backend implementations ──────────────────────────────────────────


def _downscale_for_surya(img):
    """Cap width per Surya troubleshooting (quality / memory on huge embedded images)."""
    from PIL import Image

    w, h = img.size
    if w <= _SURYA_MAX_WIDTH:
        return img
    scale = _SURYA_MAX_WIDTH / w
    nh = max(1, int(h * scale))
    return img.resize((_SURYA_MAX_WIDTH, nh), Image.Resampling.LANCZOS)


def _ocr_surya(img, bundle: tuple[object, object, object]) -> str:
    """Run Surya detection + recognition on a PIL Image."""
    _, recognition, detection = bundle
    img = _downscale_for_surya(img)
    predictions = recognition([img], det_predictor=detection)
    if not predictions:
        return ""
    result = predictions[0]
    lines = getattr(result, "text_lines", None) or []
    texts = [ln.text for ln in lines if getattr(ln, "text", None)]
    return "\n".join(texts).strip()


def _ocr_tesseract(img, *, lang: str = "eng") -> str:
    """Run Tesseract on a PIL Image and return text."""
    import pytesseract

    text: str = pytesseract.image_to_string(img, lang=lang).strip()
    return text
