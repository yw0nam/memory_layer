"""Starlette REST API for memory search and storage."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import asyncpg
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from memory_base.core.config import DB_URL, require_env
from memory_base.core.logger import setup_logging
from memory_base.retrieval.decompose import DeepResult, deep_search
from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import search
from memory_base.retrieval.search import validate_search_options
from memory_base.serve import admin
from memory_base.serve import ingest_api
from memory_base.serve import repos
from memory_base.serve.access_log import log_retrieval
from memory_base.serve.notes import save_note

TEXT_LIMIT = 2000
SOURCES = ("all", "code", "memory")
HEALTH_PROBE_TIMEOUT_SECONDS = 5.0


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


async def _models_endpoint_healthy(env_var: str) -> bool:
    """Return whether the vLLM server's /models path answers 2xx, without running inference."""
    base_url = require_env(env_var).rstrip("/")
    async with httpx.AsyncClient(timeout=HEALTH_PROBE_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{base_url}/models")
    return 200 <= response.status_code < 300


async def embedding_healthy() -> bool:
    """Return whether the embedding endpoint (EMB_URL) is reachable."""
    return await _models_endpoint_healthy("EMB_URL")


async def rerank_healthy() -> bool:
    """Return whether the rerank endpoint (RERANK_URL) is reachable."""
    return await _models_endpoint_healthy("RERANK_URL")


async def llm_healthy() -> bool:
    """Return whether the LLM endpoint (LLM_URL) is reachable."""
    return await _models_endpoint_healthy("LLM_URL")


async def _probe(check: Callable[[], Awaitable[bool]]) -> bool:
    """Run a health probe, turning any exception into a false result."""
    try:
        return bool(await check())
    except Exception:
        return False


def _error(message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=400)


async def _json_body(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


def _ids(body: dict[str, Any]) -> list[str] | None:
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
        return None
    return ids


async def health(request: Request) -> JSONResponse:
    """Report health of the DB, embedding, rerank, and LLM dependencies."""
    del request
    db, embedding, rerank, llm = await asyncio.gather(
        _probe(db_healthy),
        _probe(embedding_healthy),
        _probe(rerank_healthy),
        _probe(llm_healthy),
    )
    checks = {"db": db, "embedding": embedding, "rerank": rerank, "llm": llm}
    required_up = db and embedding and rerank
    status_code = 200 if required_up else 503
    return JSONResponse(
        {"status": "ok" if required_up else "error", "checks": checks},
        status_code=status_code,
    )


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


def _serialize_deep_result(result: DeepResult) -> dict[str, Any]:
    evidence = []
    for entry in result.evidence:
        evidence.append(
            {
                "ref": entry.ref,
                "text": entry.text[:TEXT_LIMIT],
                "kind": entry.kind,
                "tags": entry.tags,
                "date": datetime.fromtimestamp(entry.date, tz=timezone.utc).strftime("%Y-%m-%d"),
                "hop": entry.hop,
                "atom_question": entry.atom_question,
                "id": entry.id,
            }
        )
    trace = []
    for entry in result.trace:
        trace.append(
            {
                "hop": entry.hop,
                "sub_questions": entry.sub_questions,
                "selected_ref": entry.selected_ref,
            }
        )
    return {
        "evidence": evidence,
        "trace": trace,
        "hops_used": result.hops_used,
        "stopped_reason": result.stopped_reason,
    }


async def deep_search_route(request: Request) -> JSONResponse:
    """Validate and execute a deep search request."""
    try:
        body = await _json_body(request)
    except Exception as exc:
        return _error(f"invalid JSON body: {exc}")

    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return _error("query must be a non-empty string")

    include_archived = body.get("include_archived", False)
    if not isinstance(include_archived, bool):
        return _error("include_archived must be a boolean")

    if "tags" in body and body["tags"] is None:
        return _error("tags must be a non-empty list of strings")

    try:
        result = await deep_search(
            query,
            max_hops=body.get("max_hops"),
            kind=body.get("kind"),
            tags=body.get("tags"),
            include_archived=include_archived,
        )
    except ValueError as exc:
        return _error(str(exc))

    evidence_hits = [
        Hit(
            source="memory",
            ref=entry.ref,
            text=entry.text,
            ts=entry.date,
            meta={"id": entry.id},
        )
        for entry in result.evidence
    ]
    await log_retrieval(query, "memory", evidence_hits)

    return JSONResponse(_serialize_deep_result(result))


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
    rows = await admin.list_old_notes(older_than_days)
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
    rows = await admin.notes_by_ids(ids)
    if {row["id"] for row in rows} != set(ids):
        return _error("ids must refer only to agent_note rows")
    if body.get("confirm") is True:
        deleted = await admin.delete_notes(ids)
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
    pairs = await admin.find_duplicates(threshold, kind, limit)
    return JSONResponse({"pairs": pairs})


async def admin_archive_route(request: Request) -> JSONResponse:
    """Preview or archive cold memory rows."""
    try:
        body = await _json_body(request)
    except Exception as exc:
        return _error(f"invalid JSON body: {exc}")
    now = time.time()
    candidates = await admin.archive_candidates(now)
    if body.get("confirm") is True:
        archived = await admin.archive_rows([row["id"] for row in candidates], now)
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
        restored = await admin.restore_rows(ids)
        return JSONResponse({"restored": restored})
    rows = await admin.rows_by_ids(ids)
    return JSONResponse({"rows": rows})


setup_logging()

app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/search", search_route, methods=["POST"]),
        Route("/search/deep", deep_search_route, methods=["POST"]),
        Route("/save_memory", save_memory_route, methods=["POST"]),
        Route("/ingest/document", ingest_api.ingest_document_route, methods=["POST"]),
        Route("/ingest/jobs/{job_id}", ingest_api.ingest_job_route, methods=["GET"]),
        Route("/repos", repos.ingest_repo_route, methods=["POST"]),
        Route("/repos", repos.list_repos_route, methods=["GET"]),
        Route("/repos/jobs/{job_id}", repos.repo_job_route, methods=["GET"]),
        Route("/repos/{name}", repos.remove_repo_route, methods=["DELETE"]),
        Route("/admin/notes", admin_notes_route, methods=["GET"]),
        Route("/admin/notes/delete", admin_notes_delete_route, methods=["POST"]),
        Route("/admin/duplicates", admin_duplicates_route, methods=["GET"]),
        Route("/admin/archive", admin_archive_route, methods=["POST"]),
        Route("/admin/restore", admin_restore_route, methods=["POST"]),
    ]
)
