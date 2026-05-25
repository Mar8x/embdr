"""
MCP server for embd — exposes the retrieval API as a tool for Claude CLI and Claude Desktop.

Configuration (via environment variables, set in your Claude config file):
  EMBD_BASE_URL  Base URL of the embd retrieval server (required)
  EMBD_API_KEY   Bearer token for authentication (required)

Run directly:
  embd-mcp                         # recommended: standalone entry point
  embd mcp-serve                   # alternative: via the embd CLI (requires CWD with config.toml)
  EMBD_BASE_URL=... EMBD_API_KEY=... embd-mcp   # inline for testing

See docs/mcp.md for Claude CLI and Claude Desktop setup.
"""
from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "embd",
    instructions=(
        "Search personal documents indexed in an embd retrieval server. "
        "Use search_documents for any question that may be answered by your private document library. "
        "Results include the source filename, page number, and a relevance score."
    ),
)


def _base_url() -> str:
    url = os.environ.get("EMBD_BASE_URL", "").rstrip("/")
    if not url:
        raise RuntimeError(
            "EMBD_BASE_URL is not set. "
            "Set it to your embd server base URL (e.g. https://embd.example.com) "
            "in your Claude MCP server config."
        )
    return url


def _api_key() -> str:
    key = os.environ.get("EMBD_API_KEY", "")
    if not key:
        raise RuntimeError(
            "EMBD_API_KEY is not set. "
            "Set it to the Bearer token configured on your embd server "
            "in your Claude MCP server config."
        )
    return key


@mcp.tool()
def search_documents(
    query: str,
    source_id: str | None = None,
    top_k: int = 5,
) -> str:
    """Search the personal document index for passages relevant to a query.

    Args:
        query: Natural-language question or search phrase.
        source_id: Optional — restrict results to one document by its source key
                   (e.g. "report.pdf" or "papers/report.pdf").
        top_k: Number of passages to return (1–20, default 5).

    Returns cited passages in the form [source p.N] (score X.XX) followed by the text.
    """
    payload: dict = {
        "queries": [
            {
                "query": query,
                "top_k": max(1, min(top_k, 20)),
                **({"filter": {"source_id": source_id}} if source_id else {}),
            }
        ]
    }

    try:
        r = httpx.post(
            f"{_base_url()}/query",
            headers={"Authorization": f"Bearer {_api_key()}"},
            json=payload,
            timeout=30.0,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            return "Authentication failed — check EMBD_API_KEY."
        if exc.response.status_code == 403:
            return "Access denied — your IP may not be in the server's allowlist."
        return f"Server error {exc.response.status_code}: {exc.response.text[:200]}"
    except httpx.RequestError as exc:
        return f"Could not reach embd server ({_base_url()}): {exc}"

    chunks = r.json()["results"][0]["results"]
    if not chunks:
        return "No relevant passages found."

    parts: list[str] = []
    for c in chunks:
        m = c["metadata"]
        ref = m["source_id"]
        if m.get("page_num"):
            ref += f" p.{m['page_num']}"
        parts.append(f"[{ref}] (score {c['score']:.2f})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def main() -> None:
    """Entry point for the `embd-mcp` script."""
    mcp.run()


if __name__ == "__main__":
    main()
