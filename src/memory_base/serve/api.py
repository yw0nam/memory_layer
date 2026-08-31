"""Starlette REST API for memory search and storage."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from memory_base.core import db
from memory_base.core.config import require_env
from memory_base.core.logger import setup_logging
from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import UpstreamUnavailable
from memory_base.retrieval.search import normalize_namespaces
from memory_base.retrieval.search import search
from memory_base.serve import access_log
from memory_base.serve import admin
from memory_base.serve import ingest_api
from memory_base.serve import job_store
from memory_base.serve import namespaces
from memory_base.serve import notes
from memory_base.serve import repos
from memory_base.serve import tables
from memory_base.serve.auth import ApiKeyAuthMiddleware
from memory_base.serve.http import TEXT_LIMIT
from memory_base.serve.http import error
from memory_base.serve.http import json_body
from memory_base.serve.notes import save_note

SOURCES = ("all", "code", "memory")
# Beyond this a query is a pasted payload, not a question: it costs embedder and BM25
# work no ranking can use.
MAX_QUERY_CHARS = 2000
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
    repo = hit.meta.get("repo")
    if repo:
        out["repo"] = repo
    context = hit.meta.get("context")
    if context:
        out["context"] = context
    if hit.meta.get("archived"):
        out["archived"] = True
    if "columns" in hit.meta:
        out["columns"] = hit.meta["columns"]
    return out


async def db_healthy() -> bool:
    """Return whether the configured database accepts a simple query."""
    async with db.acquire(timeout=HEALTH_PROBE_TIMEOUT_SECONDS) as conn:
        return bool(await conn.fetchval("SELECT 1"))


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


def _ids(body: dict[str, Any]) -> list[str] | None:
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
        return None
    return ids


async def health(request: Request) -> JSONResponse:
    """Report that the process serves HTTP, reaching nothing outside it."""
    del request
    return JSONResponse({"status": "ok"})


async def health_services(request: Request) -> JSONResponse:
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
    """Validate and execute a hybrid search request, scoped to the caller's allowed namespaces."""
    key = request.state.key
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")

    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return error("query must be a non-empty string")
    query = query[:MAX_QUERY_CHARS]
    source = body.get("source", "all")
    if source not in SOURCES:
        return error(f"source must be one of {SOURCES}")
    top_k = body.get("top_k", 10)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        return error("top_k must be an integer")

    include_archived = body.get("include_archived", False)
    if not isinstance(include_archived, bool):
        return error("include_archived must be a boolean")

    if "min_score" in body:
        min_score = body["min_score"]
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
            return error("min_score must be a number between 0 and 1")
        if not 0 <= min_score <= 1:
            return error("min_score must be a number between 0 and 1")

    # Explicit null is a caller error distinct from an omitted key; search() sees
    # only the resolved value and cannot tell the two apart, so this stays here.
    if "tags" in body and body["tags"] is None:
        return error("tags must be a non-empty list of strings")
    if "repo" in body and body["repo"] is None:
        return error("repo must be a non-empty list of strings")
    if "since" in body and body["since"] is None:
        return error("since must be an ISO 8601 date or datetime string")
    if "until" in body and body["until"] is None:
        return error("until must be an ISO 8601 date or datetime string")
    try:
        requested_namespaces = normalize_namespaces(body.get("namespaces"))
        if requested_namespaces is not None and not key.permits_all(set(requested_namespaces)):
            return error("requested namespaces are outside the caller's allowed set", 403)
        options: dict[str, Any] = {
            "source": source,
            "include_archived": include_archived,
        }
        if "kind" in body:
            options["kind"] = body["kind"]
        if "tags" in body:
            options["tags"] = body["tags"]
        if "repo" in body:
            options["repo"] = body["repo"]
        if "since" in body:
            options["since"] = body["since"]
        if "until" in body:
            options["until"] = body["until"]
        if "min_score" in body:
            options["min_score"] = body["min_score"]
        if requested_namespaces is not None:
            options["namespaces"] = requested_namespaces
        elif not key.is_admin:
            options["namespaces"] = sorted(key.allowed)
        hits = (await search(query, **options))[:top_k]
    except UpstreamUnavailable as exc:
        return error(f"search unavailable: {exc}, so memory cannot be attached right now", 503)
    except ValueError as exc:
        return error(str(exc))
    access_log.record_retrieval(query, source, hits)
    return JSONResponse([hit_to_dict(hit) for hit in hits])


async def save_memory_route(request: Request) -> JSONResponse:
    """Validate and store an agent-authored memory request; omitted namespace lands in key.home."""
    key = request.state.key
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")

    namespace = body.get("namespace", key.home)
    if not isinstance(namespace, str) or not namespace.strip():
        return error("namespace must be a non-empty string")
    if not key.permits(namespace):
        return error(f"namespace {namespace!r} is outside the caller's allowed set", 403)
    if "occurred_at" in body and body["occurred_at"] is None:
        return error("occurred_at must be an ISO 8601 date or datetime string")
    try:
        result = await save_note(
            body.get("content", ""),
            kind=body.get("kind", "note"),
            tags=body.get("tags"),
            supersedes=body.get("supersedes"),
            namespace=namespace,
            occurred_at=body.get("occurred_at"),
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(result)


async def notes_list_route(request: Request) -> JSONResponse:
    """List agent notes by filters alone — no query, no embedding — scoped like search."""
    key = request.state.key
    params = request.query_params
    try:
        limit = int(params.get("limit", str(notes.LIST_NOTES_DEFAULT_LIMIT)))
    except ValueError:
        return error("limit must be an integer")
    include_archived = params.get("include_archived", "false").lower()
    if include_archived not in ("true", "false"):
        return error("include_archived must be true or false")
    try:
        requested_namespaces = normalize_namespaces(params.getlist("namespace") or None)
    except ValueError as exc:
        return error(str(exc))
    if requested_namespaces is not None and not key.permits_all(set(requested_namespaces)):
        return error("requested namespaces are outside the caller's allowed set", 403)
    scope = requested_namespaces
    if scope is None and not key.is_admin:
        scope = sorted(key.allowed)
    try:
        rows = await notes.list_notes(
            tags=params.getlist("tags") or None,
            kind=params.get("kind") or None,
            namespaces=scope,
            include_archived=include_archived == "true",
            since=params.get("since"),
            until=params.get("until"),
            limit=limit,
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(rows)


def _admin_scope(key) -> list[str] | None:
    """None means unfiltered (an admin key); otherwise the caller's allowed set."""
    return None if key.is_admin else sorted(key.allowed)


async def admin_notes_route(request: Request) -> JSONResponse:
    """List active agent notes older than a requested age, scoped to the caller's namespaces."""
    try:
        older_than_days = int(request.query_params.get("older_than_days", "90"))
    except ValueError:
        return error("older_than_days must be an integer")
    rows = await admin.list_old_notes(older_than_days, namespaces=_admin_scope(request.state.key))
    return JSONResponse(rows)


async def admin_notes_delete_route(request: Request) -> JSONResponse:
    """Preview or delete selected agent notes, scoped to the caller's namespaces."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    ids = _ids(body)
    if ids is None:
        return error("ids must be a non-empty list")
    scope = _admin_scope(request.state.key)
    rows = await admin.notes_by_ids(ids, namespaces=scope)
    if {row["id"] for row in rows} != set(ids):
        return error("ids must refer only to agent_note rows")
    if body.get("confirm") is True:
        deleted = await admin.delete_notes(ids, namespaces=scope)
        return JSONResponse({"deleted": deleted})
    return JSONResponse({"rows": rows})


async def admin_notes_move_route(request: Request) -> JSONResponse:
    """Move agent notes into another registered namespace; admin keys only."""
    key = request.state.key
    if not key.is_admin:
        return error("admin key required", 403)
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    ids = _ids(body)
    if ids is None:
        return error("ids must be a non-empty list")
    target_namespace = body.get("namespace")
    if not isinstance(target_namespace, str) or not target_namespace.strip():
        return error("namespace must be a non-empty string")
    try:
        result = await admin.move_notes(ids, target_namespace)
    except namespaces.NamespaceError as exc:
        return error(str(exc))
    return JSONResponse(result)


async def admin_duplicates_route(request: Request) -> JSONResponse:
    """List active near-duplicate memory pairs, scoped to the caller's namespaces."""
    try:
        threshold = float(request.query_params.get("threshold", "0.9"))
    except ValueError:
        return error("threshold must be a number")
    kind = request.query_params.get("kind") or None
    try:
        limit = int(request.query_params.get("limit", "50"))
    except ValueError:
        return error("limit must be an integer")
    pairs = await admin.find_duplicates(
        threshold, kind, limit, namespaces=_admin_scope(request.state.key)
    )
    return JSONResponse({"pairs": pairs})


async def admin_archive_route(request: Request) -> JSONResponse:
    """Preview or archive cold memory rows, scoped to the caller's namespaces."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    now = time.time()
    scope = _admin_scope(request.state.key)
    candidates = await admin.archive_candidates(now, namespaces=scope)
    if body.get("confirm") is True:
        archived = await admin.archive_rows(
            [row["id"] for row in candidates], now, namespaces=scope
        )
        return JSONResponse({"archived": archived})
    return JSONResponse({"candidates": candidates})


async def admin_restore_route(request: Request) -> JSONResponse:
    """Preview or restore selected memory rows, scoped to the caller's namespaces."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    ids = _ids(body)
    if ids is None:
        return error("ids must be a non-empty list")
    scope = _admin_scope(request.state.key)
    if body.get("confirm") is True:
        restored = await admin.restore_rows(ids, namespaces=scope)
        return JSONResponse({"restored": restored})
    rows = await admin.rows_by_ids(ids, namespaces=scope)
    return JSONResponse({"rows": rows})


async def namespaces_create_route(request: Request) -> JSONResponse:
    """Register a new namespace; 400 on a bad slug, 409 on a duplicate name.

    A private namespace records the caller's key label as owner.
    """
    key = request.state.key
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    visibility = body.get("visibility", "public")
    owner = key.label if visibility == "private" else None
    try:
        result = await namespaces.create_namespace(body.get("name"), visibility, owner)
    except namespaces.NamespaceExistsError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except namespaces.NamespaceError as exc:
        return error(str(exc))
    return JSONResponse(result, status_code=201)


async def namespaces_list_route(request: Request) -> JSONResponse:
    """List the caller's allowed namespaces (every namespace for an admin key)."""
    key = request.state.key
    rows = await namespaces.list_namespaces()
    if not key.is_admin:
        rows = [row for row in rows if row["name"] in key.allowed]
    return JSONResponse(rows)


async def namespaces_delete_route(request: Request) -> JSONResponse:
    """Unregister a namespace: 400 reserved, 404 unknown, 403 non-owner, 409 non-empty."""
    key = request.state.key
    name = request.path_params["name"]
    if name == namespaces.DEFAULT_NAMESPACE:
        return error("the 'default' namespace is reserved and cannot be deleted")
    ns = await namespaces.get_namespace(name)
    if ns is None:
        return JSONResponse({"error": f"unknown namespace: {name}"}, status_code=404)
    if not (key.is_admin or ns["owner"] == key.label):
        return error(f"not permitted to delete namespace: {name}", 403)
    try:
        await namespaces.delete_namespace(name)
    except namespaces.NamespaceReservedError as exc:
        return error(str(exc))
    except namespaces.NamespaceNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except namespaces.NamespaceNotEmptyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse({"deleted": name})


setup_logging()


class HealthAccessFilter(logging.Filter):
    """Successful liveness probes drown real requests in the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            _, method, path, _, status = record.args
        except (TypeError, ValueError):
            return True
        return not (method == "GET" and path == "/health" and status == 200)


logging.getLogger("uvicorn.access").addFilter(HealthAccessFilter())


@asynccontextmanager
async def lifespan(app: Starlette):
    """Recover durable jobs, run workers and the hit flusher, and close the pool last."""
    del app
    await job_store.initialize()
    workers = job_store.start_workers()
    flusher = access_log.start_flusher()
    try:
        yield
    finally:
        await access_log.stop_flusher(flusher)
        await job_store.stop_workers(workers)
        await db.close_table_query_pool()
        await db.close_pool()


app = Starlette(
    lifespan=lifespan,
    middleware=[Middleware(ApiKeyAuthMiddleware)],
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/health/services", health_services, methods=["GET"]),
        Route("/search", search_route, methods=["POST"]),
        Route("/save_memory", save_memory_route, methods=["POST"]),
        Route("/notes", notes_list_route, methods=["GET"]),
        Route("/tables/query", tables.tables_query_route, methods=["POST"]),
        Route("/ingest/document", ingest_api.ingest_document_route, methods=["POST"]),
        Route("/ingest/jobs", ingest_api.ingest_jobs_route, methods=["GET"]),
        Route("/ingest/jobs/{job_id}", ingest_api.ingest_job_route, methods=["GET"]),
        Route(
            "/ingest/documents/{document_id}",
            ingest_api.remove_document_route,
            methods=["DELETE"],
        ),
        Route("/repos", repos.ingest_repo_route, methods=["POST"]),
        Route("/repos", repos.list_repos_route, methods=["GET"]),
        Route("/repos/jobs/{job_id}", repos.repo_job_route, methods=["GET"]),
        Route("/repos/{name}", repos.remove_repo_route, methods=["DELETE"]),
        Route("/namespaces", namespaces_create_route, methods=["POST"]),
        Route("/namespaces", namespaces_list_route, methods=["GET"]),
        Route("/namespaces/{name}", namespaces_delete_route, methods=["DELETE"]),
        Route("/admin/notes", admin_notes_route, methods=["GET"]),
        Route("/admin/notes/delete", admin_notes_delete_route, methods=["POST"]),
        Route("/admin/notes/move", admin_notes_move_route, methods=["POST"]),
        Route("/admin/duplicates", admin_duplicates_route, methods=["GET"]),
        Route("/admin/archive", admin_archive_route, methods=["POST"]),
        Route("/admin/restore", admin_restore_route, methods=["POST"]),
    ],
)
