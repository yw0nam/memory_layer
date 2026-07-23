"""Starlette REST API for memory search and storage."""

from __future__ import annotations

import inspect
import time
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
from memory_base.retrieval.search import validate_search_options
from memory_base.serve import admin
from memory_base.serve import ingest_api
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
        "score": hit.score,
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


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _ids(body: dict[str, Any]) -> list[str] | None:
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
        return None
    return ids


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

    include_archived = body.get("include_archived", False)
    if not isinstance(include_archived, bool):
        return _error("include_archived must be a boolean")

    include_atoms = body.get("include_atoms")
    if "include_atoms" in body and not isinstance(include_atoms, bool):
        return _error("include_atoms must be a boolean")
    if "tags" in body and body["tags"] is None:
        return _error("tags must be a non-empty list of strings")
    try:
        kind, tags = validate_search_options(source, body.get("kind"), body.get("tags"))
        options: dict[str, Any] = {
            "source": source,
            "include_archived": include_archived,
        }
        if "kind" in body:
            options["kind"] = kind
        if "tags" in body:
            options["tags"] = tags
        if "include_atoms" in body:
            options["include_atoms"] = include_atoms
        hits = (await search(query, **options))[:top_k]
    except ValueError as exc:
        return _error(str(exc))
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
            supersedes=body.get("supersedes"),
        )
    except ValueError as exc:
        return _error(str(exc))
    return JSONResponse(result)


async def admin_notes_route(request: Request) -> JSONResponse:
    """List active agent notes older than a requested age."""
    try:
        older_than_days = int(request.query_params.get("older_than_days", "90"))
    except ValueError:
        return _error("older_than_days must be an integer")
    rows = await _resolve(admin.list_old_notes(older_than_days))
    return JSONResponse(rows)


async def admin_notes_delete_route(request: Request) -> JSONResponse:
    """Preview or delete selected agent notes."""
    try:
        body = await _json_body(request)
    except Exception as exc:
        return _error(f"invalid JSON body: {exc}")
    ids = _ids(body)
    if ids is None:
        return _error("ids must be a non-empty list")
    rows = await _resolve(admin.notes_by_ids(ids))
    if {row["id"] for row in rows} != set(ids):
        return _error("ids must refer only to agent_note rows")
    if body.get("confirm") is True:
        deleted = await _resolve(admin.delete_notes(ids))
        return JSONResponse({"deleted": deleted})
    return JSONResponse({"rows": rows})


async def admin_duplicates_route(request: Request) -> JSONResponse:
    """List active near-duplicate memory pairs."""
    try:
        threshold = float(request.query_params.get("threshold", "0.9"))
    except ValueError:
        return _error("threshold must be a number")
    kind = request.query_params.get("kind") or None
    try:
        limit = int(request.query_params.get("limit", "50"))
    except ValueError:
        return _error("limit must be an integer")
    pairs = await _resolve(admin.find_duplicates(threshold, kind, limit))
    return JSONResponse({"pairs": pairs})


async def admin_archive_route(request: Request) -> JSONResponse:
    """Preview or archive cold memory rows."""
    try:
        body = await _json_body(request)
    except Exception as exc:
        return _error(f"invalid JSON body: {exc}")
    now = time.time()
    candidates = await _resolve(admin.archive_candidates(now))
    if body.get("confirm") is True:
        archived = await _resolve(admin.archive_rows([row["id"] for row in candidates], now))
        return JSONResponse({"archived": archived})
    return JSONResponse({"candidates": candidates})


async def admin_restore_route(request: Request) -> JSONResponse:
    """Preview or restore selected memory rows."""
    try:
        body = await _json_body(request)
    except Exception as exc:
        return _error(f"invalid JSON body: {exc}")
    ids = _ids(body)
    if ids is None:
        return _error("ids must be a non-empty list")
    if body.get("confirm") is True:
        restored = await _resolve(admin.restore_rows(ids))
        return JSONResponse({"restored": restored})
    rows = await _resolve(admin.rows_by_ids(ids))
    return JSONResponse({"rows": rows})


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/search", search_route, methods=["POST"]),
        Route("/save_memory", save_memory_route, methods=["POST"]),
        Route("/ingest/document", ingest_api.ingest_document_route, methods=["POST"]),
        Route("/ingest/jobs/{job_id}", ingest_api.ingest_job_route, methods=["GET"]),
        Route("/admin/notes", admin_notes_route, methods=["GET"]),
        Route("/admin/notes/delete", admin_notes_delete_route, methods=["POST"]),
        Route("/admin/duplicates", admin_duplicates_route, methods=["GET"]),
        Route("/admin/archive", admin_archive_route, methods=["POST"]),
        Route("/admin/restore", admin_restore_route, methods=["POST"]),
    ]
)
