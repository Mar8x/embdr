"""
OpenAI retrieval-style HTTP API for semantic search over the local vector index.

Compatible with ChatGPT custom GPT Actions that use API key (Bearer) auth and
a POST /query endpoint. See https://github.com/openai/chatgpt-retrieval-plugin
for the original schema.
"""
from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from .config import Config
from .embedding.encoder import Encoder
from .store.vector_store import K_DOC_DATE, K_DOC_DATE_Q, K_PAGE, K_SOURCE, VectorStore

logger = logging.getLogger(__name__)


def _absolute_server_url_for_openapi(request: Request, cfg: Config) -> str:
    """Absolute `servers[0].url` for OpenAPI (ChatGPT rejects relative `/`)."""
    explicit = (cfg.server.openapi_base_url or "").strip().rstrip("/")
    if explicit:
        return explicit
    proto = request.headers.get("x-forwarded-proto")
    if proto:
        proto = proto.split(",")[0].strip()
    else:
        proto = request.url.scheme
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").strip()
    if host:
        host = host.split(",")[0].strip()
    if host and proto:
        return f"{proto}://{host}".rstrip("/")
    return ""


def resolve_api_key(cfg: Config) -> str:
    """Prefer EMBD_API_KEY env over config.toml ``[server].api_key``."""
    env = (os.environ.get("EMBD_API_KEY") or "").strip()
    if env:
        return env
    return (cfg.server.api_key or "").strip()


class DocumentMetadataFilter(BaseModel):
    """Optional filter to scope results to a single indexed document."""

    model_config = ConfigDict(extra="ignore")

    source_id: str | None = Field(
        default=None,
        description=(
            "Exact document filename to restrict search to (e.g. `report.pdf`). "
            "Must match a `metadata.source_id` value from a previous result. "
            "Omit to search all documents."
        ),
        examples=["handbook.pdf"],
    )
    document_id: str | None = Field(
        default=None,
        description="Alias for `source_id`. If both are set, `source_id` takes precedence.",
    )


class QueryItem(BaseModel):
    """One search to run. Each item is searched independently and produces one result group."""

    query: str = Field(
        ...,
        description=(
            "Natural-language search text. Write a focused, specific search string for "
            "best results. Split broad questions into multiple targeted sub-queries."
        ),
        examples=["What is the Trail of Bits agent strategy?"],
    )
    filter: DocumentMetadataFilter | None = Field(
        default=None,
        description=(
            "Optional: limit results to a single indexed document by filename. "
            "Use an exact `source_id` from a previous result."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description=(
            "Maximum passages to return for this sub-query. Omit to use the server default. "
            "Use 3\u20135 for precise lookups, 8\u201312 for broad overviews."
        ),
        examples=[5],
    )


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "queries": [
                        {
                            "query": "What is the agent strategy?",
                            "top_k": 5,
                        },
                        {
                            "query": "Summarize the deployment section",
                            "filter": {"source_id": "runbook.pdf"},
                            "top_k": 3,
                        },
                    ]
                }
            ]
        }
    )

    queries: list[QueryItem] = Field(
        ...,
        description=(
            "One or more searches to run in a single request. Each entry produces one element "
            "in the response `results` array, in the same order. Split complex questions into "
            "focused sub-queries for better retrieval."
        ),
    )


class RetrievedChunkMetadata(BaseModel):
    """Provenance metadata for a retrieved passage. Use `source_id` and `page_num` for inline citations."""

    source: Literal["file"] = Field(
        "file",
        description="Always `file`.",
    )
    source_id: str = Field(
        ...,
        description=(
            "Exact document name (e.g. `report.pdf`). Use this verbatim in citations as "
            "[source_id, p.N]. Also use as `filter.source_id` to search only this document "
            "in follow-up queries."
        ),
    )
    document_id: str = Field(
        ...,
        description="Same value as `source_id`. Use `source_id` for citations.",
    )
    page_num: int | None = Field(
        default=None,
        description=(
            "1-based page number where this passage appears. Null for unpaginated sources "
            "(e.g. web pages, plain text). Include in citations as `p.N` when present."
        ),
    )
    document_date: str | None = Field(
        default=None,
        description=(
            "Best-effort publication or creation date of the source document (ISO-8601, "
            "e.g. `2024-03-15T00:00:00+00:00`). Null when unknown. When passages from "
            "different documents conflict, prefer the one with the more recent and "
            "higher-quality date."
        ),
    )
    document_date_quality: str | None = Field(
        default=None,
        description=(
            "How reliable `document_date` is: "
            "`document_metadata` = extracted from the file's own metadata (most trustworthy), "
            "`filesystem` = OS file modification time (less reliable), "
            "`none` = no date available. "
            "Use this to weigh conflicting information from sources with different dates."
        ),
    )


class DocumentChunkWithScore(BaseModel):
    """One retrieved passage with its provenance metadata and relevance score."""

    id: str | None = Field(default=None, description="Internal chunk identifier. Not needed for citations.")
    text: str = Field(
        ...,
        description=(
            "The passage text. Read this to find evidence, then cite the source using "
            "`metadata.source_id` and `metadata.page_num`."
        ),
    )
    metadata: RetrievedChunkMetadata
    score: float = Field(
        ...,
        description=(
            "Relative relevance score in [0, 1] \u2014 higher means more relevant. Use to "
            "prioritize which passages to read first. Do not treat the absolute number as a "
            "confidence measure; compare ranks across chunks instead."
        ),
        examples=[0.82],
    )
    embedding: list[float] | None = Field(
        default=None,
        description="Not populated; reserved for schema compatibility.",
    )


class QueryResultItem(BaseModel):
    """One result group: the echoed query plus its matching passages."""

    query: str = Field(..., description="Echo of the corresponding request query string.")
    results: list[DocumentChunkWithScore] = Field(
        ...,
        description="Matching passages for this sub-query, ordered by relevance (best first).",
    )


class QueryResponse(BaseModel):
    """Response: one result group per input query, same length and order."""

    results: list[QueryResultItem] = Field(
        ...,
        description="One result object per request query, in the same order as the request.",
    )


class HealthResponse(BaseModel):
    """Liveness response (strict schema for OpenAPI validators)."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = Field(
        default="ok",
        description="Always `ok` when the HTTP server is running.",
    )


class RootResponse(BaseModel):
    """Root discovery payload."""

    model_config = ConfigDict(extra="forbid")

    service: Literal["embd-retrieval"] = Field(
        default="embd-retrieval",
        description="Service identifier.",
    )
    docs: str = Field(default="/docs", description="Relative path to Swagger UI.")
    redoc: str = Field(default="/redoc", description="Relative path to ReDoc.")
    openapi_json: str = Field(
        default="/openapi.json",
        description="Relative path to this OpenAPI document.",
    )
    query: str = Field(
        default="POST /query (Bearer auth required)",
        description="How to call semantic search.",
    )


def _distance_to_score(distance: float) -> float:
    """Chroma cosine space: distance ≈ 1 − cosine similarity; map to [0, 1]."""
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - d))


def _filter_to_chroma_where(
    filter_obj: DocumentMetadataFilter | None,
) -> dict[str, Any] | None:
    """Map filter to Chroma ``where`` (indexed filename only)."""
    if filter_obj is None:
        return None
    fname = filter_obj.source_id or filter_obj.document_id
    if fname:
        return {K_SOURCE: str(fname)}
    return None


def create_app(cfg: Config) -> FastAPI:
    api_key = resolve_api_key(cfg)
    if not api_key:
        raise ValueError(
            "Retrieval API requires an API key. Set EMBD_API_KEY or [server].api_key in config.toml."
        )

    bearer_scheme = HTTPBearer()

    def verify_api_key(
        creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    ) -> None:
        if len(creds.credentials) != len(api_key) or not secrets.compare_digest(
            creds.credentials, api_key
        ):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .ingestion.watcher import start_watcher

        from .ingestion.bm25_index import BM25Index, BM25_FILENAME

        logger.info("Loading embedding model (warming up) …")
        encoder = Encoder(cfg.embedding.model_name, cfg.embedding.device)
        _ = encoder._model  # force model load now so first request is fast
        store = VectorStore(cfg.paths.db_dir, cfg.retrieval.collection_name)
        bm25 = BM25Index.load(cfg.paths.db_dir / BM25_FILENAME)
        app.state.encoder = encoder
        app.state.store = store
        app.state.bm25 = bm25
        if bm25:
            logger.info("BM25 index loaded — hybrid search enabled")
        logger.info(
            "Retrieval API ready. Collection=%s chunks=%d",
            cfg.retrieval.collection_name,
            store.count(),
        )

        observer = start_watcher(store, encoder, cfg)
        logger.info(
            "File watcher active on %s — new/changed/deleted documents "
            "are auto-ingested (queries see updates immediately).",
            cfg.paths.documents_dir,
        )
        try:
            yield
        finally:
            observer.stop()
            observer.join(timeout=5)

    _API_DESCRIPTION = """\
## Purpose

**embd** is a document retrieval tool. Use it as your **primary source of evidence** before \
answering factual questions. Call `POST /query` with one or more natural-language searches; \
the API returns the most relevant text passages from the user's private document index.

## Authentication

Send header `Authorization: Bearer <API key>`.

## How to call (`POST /query`)

- **`queries`**: array of independent searches. Send one item per distinct information need; \
split complex questions into focused sub-queries.
- **`query`**: the natural-language search string.
- **`top_k`**: how many passages to return per sub-query (omit for server default).
- **`filter`**: optional. Set **`source_id`** to an exact **filename** returned in a previous \
`metadata.source_id` (e.g. `report.pdf`) to search only that document. Omit to search the \
entire collection.

## How to use the response

Each result group echoes the **`query`** and contains **`results`**: passages with:
- **`text`** — the passage to read and cite.
- **`metadata.source_id`** — the exact document name. Copy it verbatim into citations.
- **`metadata.page_num`** — 1-based page number (null for unpaginated sources). \
Use for `[source_id, p.N]` citations.
- **`metadata.document_date`** / **`document_date_quality`** — when the source was published \
or last modified, with a reliability rating. Use to prefer newer sources when passages conflict.
- **`score`** — relative relevance hint (higher is better). Use to prioritize reading order; \
do not treat the absolute number as precise.

## Citation rule

Every factual claim drawn from a passage **must** carry an inline citation using the exact \
`source_id` and `page_num` returned in `metadata`.
"""

    app = FastAPI(
        title="embd Retrieval API",
        description=_API_DESCRIPTION,
        version="0.2.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/docs", include_in_schema=False)
    async def swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{app.title} – docs",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url="/openapi.json",
            title=f"{app.title} – ReDoc",
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness check (no auth) for reverse proxies and tunnels."""
        return HealthResponse()

    @app.get("/", response_model=RootResponse)
    async def root() -> RootResponse:
        return RootResponse()

    @app.post(
        "/query",
        response_model=QueryResponse,
        dependencies=[Depends(verify_api_key)],
        operation_id="query_query_post",
        summary="Retrieve evidence passages from the document index",
        response_description="One result group per request query, in order.",
        responses={
            401: {"description": "Missing or invalid Bearer token."},
            422: {"description": "Request body does not match the documented JSON schema."},
        },
    )
    async def query_documents(body: QueryRequest) -> QueryResponse:
        """Call before answering factual questions. Send natural-language searches;
        returns top matching passages from the user's document index. Use `filter.source_id`
        with an exact `metadata.source_id` from a prior result to search one file.
        """
        encoder: Encoder = app.state.encoder
        store: VectorStore = app.state.store
        default_k = cfg.retrieval.top_k

        if not body.queries:
            return QueryResponse(results=[])

        out: list[QueryResultItem] = []
        for item in body.queries:
            q = (item.query or "").strip()
            if not q:
                out.append(QueryResultItem(query=item.query, results=[]))
                continue

            top_k = item.top_k if item.top_k is not None else default_k
            where = _filter_to_chroma_where(item.filter)
            vec = encoder.encode_query(q)

            bm25 = getattr(app.state, "bm25", None)
            if bm25 is not None and cfg.retrieval.bm25_weight > 0 and where is None:
                from .qa.hybrid_retriever import rrf_merge
                fetch_k = top_k * 2
                semantic_hits = store.query(vec, top_k=fetch_k, where=where)
                bm25_hits = bm25.query(q, top_k=fetch_k)
                hits = rrf_merge(
                    semantic_hits, bm25_hits, store,
                    semantic_weight=cfg.retrieval.semantic_weight,
                    bm25_weight=cfg.retrieval.bm25_weight,
                    top_k=top_k,
                )
            else:
                hits = store.query(vec, top_k=top_k, where=where)

            chunks: list[DocumentChunkWithScore] = []
            for h in hits:
                meta_raw = h.get("metadata") or {}
                source_name = str(meta_raw.get(K_SOURCE, ""))
                page_raw = meta_raw.get(K_PAGE)
                page_num: int | None
                if page_raw is None:
                    page_num = None
                else:
                    try:
                        page_num = int(page_raw)
                    except (TypeError, ValueError):
                        page_num = None
                raw_date = (meta_raw.get(K_DOC_DATE) or "").strip() or None
                raw_date_q = (meta_raw.get(K_DOC_DATE_Q) or "").strip() or None
                chunks.append(
                    DocumentChunkWithScore(
                        id=h.get("id"),
                        text=h.get("text") or "",
                        metadata=RetrievedChunkMetadata(
                            source="file",
                            source_id=source_name,
                            document_id=source_name,
                            page_num=page_num,
                            document_date=raw_date,
                            document_date_quality=raw_date_q,
                        ),
                        score=_distance_to_score(h.get("distance", 1.0)),
                    )
                )
            out.append(QueryResultItem(query=item.query, results=chunks))

        return QueryResponse(results=out)

    _openapi_cache: dict[str, dict[str, Any]] = {}

    def _build_openapi_schema(server_url: str) -> dict[str, Any]:
        su = server_url.rstrip("/")
        return get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
            servers=[{"url": su, "description": "Public API base URL"}],
        )

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi_json(request: Request) -> JSONResponse:
        server_url = _absolute_server_url_for_openapi(request, cfg)
        if not server_url:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Cannot build OpenAPI servers URL. Set EMBD_OPENAPI_BASE_URL (e.g. "
                    "https://embd.example.com:8801) or fetch this document through the "
                    "reverse proxy with X-Forwarded-Proto and Host set."
                ),
            )
        if server_url not in _openapi_cache:
            _openapi_cache[server_url] = _build_openapi_schema(server_url)
        return JSONResponse(_openapi_cache[server_url])

    return app
