"""
SearXNG web search client.

Queries a self-hosted or public SearXNG instance and returns structured
results that can be used as additional context for the LLM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def searxng_search(
    query: str,
    base_url: str = "http://localhost:8080",
    max_results: int = 5,
    timeout: float = 10.0,
) -> list[SearchResult]:
    """Query a SearXNG instance and return up to max_results items."""
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
    }
    try:
        resp = httpx.get(
            f"{base_url.rstrip('/')}/search",
            params=params,
            timeout=timeout,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("SearXNG request failed: %s", exc)
        return []

    data = resp.json()
    results: list[SearchResult] = []
    for item in data.get("results", [])[:max_results]:
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
        )
    return results


def format_search_context(results: list[SearchResult]) -> str:
    """Format search results as text passages for LLM context."""
    if not results:
        return ""
    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"--- Web Result {i}: {r.title} ---\n"
            f"URL: {r.url}\n"
            f"{r.snippet}"
        )
    return "\n\n".join(parts)
