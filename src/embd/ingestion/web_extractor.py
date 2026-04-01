"""
Web page text extraction for URL ingestion.

Fetches a URL, strips HTML to plain text, and returns PageText objects
compatible with the existing chunking pipeline. Reports fetch stats
(HTTP status, content-type, raw size, extracted text size) so the caller
can surface them to the user.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .doc_date import QUALITY_DOCUMENT_METADATA, QUALITY_NONE, DocumentDate
from .extractor import PageText

logger = logging.getLogger(__name__)

_SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}

_MIN_TEXT_CHARS = 50


@dataclass
class FetchResult:
    """Outcome of a URL fetch + extraction, with diagnostic info."""
    url: str
    pages: list[PageText]
    http_status: int
    content_type: str
    raw_bytes: int
    extracted_chars: int
    title: str
    error: str | None = None
    document_date: DocumentDate | None = None

    @property
    def ok(self) -> bool:
        return len(self.pages) > 0

    def summary(self) -> str:
        if self.error:
            return f"FAILED {self.url} — {self.error}"
        status = "OK" if self.ok else "EMPTY"
        parts = [
            f"{status} {self.url}",
            f"  HTTP {self.http_status} | {self.content_type}",
            f"  raw: {self.raw_bytes:,} bytes | extracted: {self.extracted_chars:,} chars",
        ]
        if self.title:
            parts.append(f"  title: {self.title}")
        if not self.ok:
            parts.append("  WARNING: no usable text extracted (page may be JS-rendered or login-gated)")
        return "\n".join(parts)


def _extract_web_date(
    soup: BeautifulSoup, resp: httpx.Response,
) -> DocumentDate:
    """Best-effort date from HTML meta tags, then HTTP Last-Modified header."""
    _META_DATE_ATTRS = [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "date"}),
        ("meta", {"name": "DC.date"}),
        ("meta", {"name": "dcterms.date"}),
        ("meta", {"property": "og:updated_time"}),
        ("meta", {"name": "last-modified"}),
        ("time", {"itemprop": "datePublished"}),
    ]
    for tag_name, attrs in _META_DATE_ATTRS:
        tag = soup.find(tag_name, attrs=attrs)
        if tag:
            raw = (tag.get("content") or tag.get("datetime") or tag.string or "").strip()
            if raw:
                from .doc_date import _normalise_date_string
                iso = _normalise_date_string(raw)
                if iso:
                    return DocumentDate(
                        date=iso,
                        quality=QUALITY_DOCUMENT_METADATA,
                        source=f"html:{tag_name}[{next(iter(attrs.values()))}]",
                    )

    last_mod = (resp.headers.get("last-modified") or "").strip()
    if last_mod:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(last_mod)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return DocumentDate(
                date=dt.isoformat(),
                quality=QUALITY_DOCUMENT_METADATA,
                source="http:last-modified",
            )
        except Exception:
            pass

    return DocumentDate(date=None, quality=QUALITY_NONE, source="none")


def fetch_and_extract(url: str, timeout: float = 15.0) -> FetchResult:
    """Fetch a URL and extract readable text, returning full diagnostics."""
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "embd/0.1 (document-ingestion)"},
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch '%s': %s", url, exc)
        return FetchResult(
            url=url, pages=[], http_status=0,
            content_type="", raw_bytes=0, extracted_chars=0,
            title="", error=str(exc),
        )

    content_type = resp.headers.get("content-type", "")
    raw_bytes = len(resp.content)

    if "html" not in content_type and "text" not in content_type:
        logger.warning("Unsupported content-type '%s' for '%s'", content_type, url)
        return FetchResult(
            url=url, pages=[], http_status=resp.status_code,
            content_type=content_type, raw_bytes=raw_bytes,
            extracted_chars=0, title="",
            error=f"unsupported content-type: {content_type}",
        )

    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    doc_date = _extract_web_date(soup, resp)

    for tag in soup.find_all(_SKIP_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = "\n".join(line for line in text.splitlines() if line.strip())

    extracted_chars = len(text)

    if extracted_chars < _MIN_TEXT_CHARS:
        logger.warning(
            "Extracted only %d chars from '%s' (min %d). "
            "Page may be JS-rendered, login-gated, or mostly images.",
            extracted_chars, url, _MIN_TEXT_CHARS,
        )
        return FetchResult(
            url=url, pages=[], http_status=resp.status_code,
            content_type=content_type, raw_bytes=raw_bytes,
            extracted_chars=extracted_chars, title=title,
            document_date=doc_date,
        )

    logger.info(
        "Fetched '%s': %d bytes raw → %d chars text, title='%s'",
        url, raw_bytes, extracted_chars, title[:80],
    )

    pages = [PageText(page_num=1, text=text)]
    return FetchResult(
        url=url, pages=pages, http_status=resp.status_code,
        content_type=content_type, raw_bytes=raw_bytes,
        extracted_chars=extracted_chars, title=title,
        document_date=doc_date,
    )


def extract_url(url: str, timeout: float = 15.0) -> list[PageText]:
    """Fetch a URL and extract readable text paragraphs.

    Thin wrapper around fetch_and_extract for callers that only need
    the page list (backward-compatible).
    """
    result = fetch_and_extract(url, timeout=timeout)
    return result.pages


def url_to_source_name(url: str) -> str:
    """Derive a short, filesystem-safe source name from a URL."""
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    path = parsed.path.strip("/").replace("/", "_")[:60]
    if path:
        return f"{host}_{path}"
    return host
