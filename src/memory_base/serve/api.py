"""Starlette REST API for memory search and storage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import asyncpg
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from memory_base.common import DB_URL
from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import search
from memory_base.serve.access_log import log_retrieval
from memory_base.serve.notes import save_note

TEXT_LIMIT = 2000
SOURCES = ("all", "code", "history")


def hit_to_dict(hit: Hit) -> dict[str, Any]:
    """Convert a search hit into a JSON-serializable response object."""
    out: dict[str, Any] = {
        "source": hit.source,
        "ref": hit.ref,
        "date": datetime.fromtimestamp(hit.ts, tz=timezone.utc).strftime("%Y-%m-%d"),
        "score": hit.rerank_score if hit.rerank_score is not None else hit.rrf,
        "text": hit.text[:TEXT_LIMIT],
    }
    context = hit.meta.get("context")
    if context:
        out["context"] = context
    return out


async def db_healthy() -> bool:
    """Return whether the configured database accepts a simple query."""
    conn = await asyncpg.connect(DB_URL)
    try:
        return bool(await conn.fetchval("SELECT 1"))
    finally:
        await conn.close()


def _error(message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=400)


async def _json_body(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


async def health(request: Request) -> JSONResponse:
    """Report API and database health."""
    del request
    try:
        healthy = await db_healthy()
    except Exception:
        healthy = False
    if healthy is False:
        return JSONResponse({"status": "error"}, status_code=503)
    return JSONResponse({"status": "ok"})


async def search_route(request: Request) -> JSONResponse:
    """Validate and execute a hybrid search request."""
    try:
        body = await _json_body(request)
    except Exception as exc:
        return _error(f"invalid JSON body: {exc}")

    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return _error("query must be a non-empty string")
    source = body.get("source", "all")
    if source not in SOURCES:
        return _error(f"source must be one of {SOURCES}")
    top_k = body.get("top_k", 10)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        return _error("top_k must be an integer")

    hits = (await search(query, source=source))[:top_k]
    await log_retrieval(query, source, hits)
    return JSONResponse([hit_to_dict(hit) for hit in hits])


async def save_memory_route(request: Request) -> JSONResponse:
    """Validate and store an agent-authored memory request."""
    try:
        body = await _json_body(request)
    except Exception as exc:
        return _error(f"invalid JSON body: {exc}")

    try:
        result = await save_note(
            body.get("content", ""),
            kind=body.get("kind", "note"),
            tags=body.get("tags"),
        )
    except ValueError as exc:
        return _error(str(exc))
    return JSONResponse(result)


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/search", search_route, methods=["POST"]),
        Route("/save_memory", save_memory_route, methods=["POST"]),
    ]
)
