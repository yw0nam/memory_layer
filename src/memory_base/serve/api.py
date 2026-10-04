"""Starlette REST API for memory search and storage."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
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
from memory_base.core.llm import resolve_llm_provider
from memory_base.core.logger import setup_logging
from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import UpstreamUnavailable
from memory_base.retrieval.search import normalize_namespaces
from memory_base.retrieval.search import search
from memory_base.serve import access_log
from memory_base.serve.access import routes as access_routes
from memory_base.serve.access.auth import ApiKeyAuthMiddleware
from memory_base.serve.common.http import TEXT_LIMIT
from memory_base.serve.common.http import error
from memory_base.serve.common.http import json_body
from memory_base.serve.consolidation import routes as consolidation_routes
from memory_base.serve.documents import pipeline as document_pipeline
from memory_base.serve.documents import routes as document_routes
from memory_base.serve.messages import routes as message_routes
from memory_base.serve.notes import routes as note_routes
from memory_base.serve.notes.store import note_date
from memory_base.serve.profiles import routes as profile_routes
from memory_base.serve.repos import cache as repo_cache
from memory_base.serve.repos import routes as repo_routes
from memory_base.serve.tables import routes as table_routes

SOURCES = ("all", "code", "memory")
# Beyond this a query is a pasted payload, not a question: it costs embedder and BM25
# work no ranking can use.
MAX_QUERY_CHARS = 2000
MAX_BUDGET_TOKENS = 32000
HEALTH_PROBE_TIMEOUT_SECONDS = 5.0


def hit_to_dict(hit: Hit) -> dict[str, Any]:
    """Convert a search hit into a JSON-serializable response object."""
    if hit.source == "memory":
        out: dict[str, Any] = {
            "source": hit.source,
            "id": hit.meta["id"],
            "kind": hit.meta["kind"],
            "tags": hit.meta["tags"],
            "ref": hit.ref,
            "date": note_date(hit.meta["occurred_at"], hit.ts),
            "score": hit.score,
            # Memory text is bounded at write time; code chunks are not.
            "text": hit.text,
        }
    else:
        out = {
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
    if hit.meta.get("author"):
        out["author"] = hit.meta["author"]
    if hit.meta.get("supersedes") is not None:
        out["supersedes"] = hit.meta["supersedes"]
    if "columns" in hit.meta:
        out["columns"] = hit.meta["columns"]
    return out


async def db_healthy() -> bool:
    """Return whether the configured database accepts a simple query."""
    async with db.acquire(timeout=HEALTH_PROBE_TIMEOUT_SECONDS) as conn:
        return bool(await conn.fetchval("SELECT 1"))


async def _models_endpoint_healthy(env_var: str) -> bool:
    """Return whether the vLLM server's /models path answers 2xx, without running inference."""
    return await _models_endpoint_healthy_url(require_env(env_var))


async def _models_endpoint_healthy_url(base_url: str) -> bool:
    async with httpx.AsyncClient(timeout=HEALTH_PROBE_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{base_url.rstrip('/')}/models")
    return 200 <= response.status_code < 300


async def embedding_healthy() -> bool:
    """Return whether the embedding endpoint (EMB_URL) is reachable."""
    return await _models_endpoint_healthy("EMB_URL")


async def rerank_healthy() -> bool:
    """Return whether the rerank endpoint (RERANK_URL) is reachable."""
    return await _models_endpoint_healthy("RERANK_URL")


async def llm_healthy() -> bool:
    """Hosted chat APIs need no probe; the vLLM fallback is probed at /models."""
    provider = resolve_llm_provider(os.environ)
    if provider.name != "vllm":
        return True
    return await _models_endpoint_healthy_url(provider.base_url)


async def _probe(check: Callable[[], Awaitable[bool]]) -> bool:
    """Run a health probe, turning any exception into a false result."""
    try:
        return bool(await check())
    except Exception:
        return False


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

    budget_tokens = body.get("budget_tokens")
    if "budget_tokens" in body and (
        isinstance(budget_tokens, bool)
        or not isinstance(budget_tokens, int)
        or not 1 <= budget_tokens <= MAX_BUDGET_TOKENS
    ):
        return error(f"budget_tokens must be an integer between 1 and {MAX_BUDGET_TOKENS}")

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
    if "author" in body and body["author"] is None:
        return error("author must be a non-empty string")
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
        if "author" in body:
            options["author"] = body["author"]
        if budget_tokens is not None:
            options["budget_tokens"] = budget_tokens
        if requested_namespaces is not None:
            options["namespaces"] = requested_namespaces
        elif not key.is_admin:
            options["namespaces"] = sorted(key.allowed)
        log_filters = {k: v for k, v in options.items() if k != "source"} | {"top_k": top_k}
        hits = await search(query, **options)
        if budget_tokens is None:
            hits = hits[:top_k]
    except UpstreamUnavailable as exc:
        return error(f"search unavailable: {exc}, so memory cannot be attached right now", 503)
    except ValueError as exc:
        return error(str(exc))
    access_log.record_retrieval(query, source, hits, filters=log_filters)
    return JSONResponse([hit_to_dict(hit) for hit in hits])


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


# Background work as (start, stop) pairs: started in order, stopped in reverse.
BACKGROUND = (
    (document_pipeline.start, document_pipeline.stop),
    (repo_cache.start, repo_cache.stop),
)


@asynccontextmanager
async def lifespan(app: Starlette):
    """Start background work and the hit flusher, stop what started in reverse, close the pools last.

    A failing start() still stops everything started before it.
    """
    del app
    async with AsyncExitStack() as stack:
        stack.push_async_callback(db.close_pool)
        stack.push_async_callback(db.close_table_query_pool)
        for start, stop in BACKGROUND:
            stack.push_async_callback(stop, await start())
        stack.push_async_callback(access_log.stop_flusher, access_log.start_flusher())
        yield


app = Starlette(
    lifespan=lifespan,
    middleware=[Middleware(ApiKeyAuthMiddleware)],
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/health/services", health_services, methods=["GET"]),
        Route("/search", search_route, methods=["POST"]),
        Route("/save_memory", note_routes.save_route, methods=["POST"]),
        Route("/notes", note_routes.list_route, methods=["GET"]),
        Route("/profiles", profile_routes.profile_route, methods=["GET"]),
        Route("/profiles/self", profile_routes.self_route, methods=["PUT"]),
        Route("/profiles/versions", profile_routes.versions_route, methods=["GET"]),
        Route("/profiles/user/proposals", profile_routes.propose_route, methods=["POST"]),
        Route("/profiles/user/proposals", profile_routes.proposals_route, methods=["GET"]),
        Route(
            "/profiles/user/proposals/{proposal_id}", profile_routes.proposal_route, methods=["GET"]
        ),
        Route(
            "/profiles/user/proposals/{proposal_id}/approve",
            profile_routes.approve_route,
            methods=["POST"],
        ),
        Route(
            "/profiles/user/proposals/{proposal_id}/reject",
            profile_routes.reject_route,
            methods=["POST"],
        ),
        Route("/messages", message_routes.send_route, methods=["POST"]),
        Route("/messages", message_routes.list_route, methods=["GET"]),
        Route("/messages/{message_id}/claim", message_routes.claim_route, methods=["POST"]),
        Route("/messages/{message_id}", message_routes.cancel_route, methods=["DELETE"]),
        Route("/tables/query", table_routes.query_route, methods=["POST"]),
        Route("/ingest/document", document_routes.ingest_route, methods=["POST"]),
        Route("/ingest/jobs", document_routes.jobs_route, methods=["GET"]),
        Route("/ingest/jobs/{job_id}", document_routes.job_route, methods=["GET"]),
        Route("/ingest/documents/{document_id}", document_routes.remove_route, methods=["DELETE"]),
        Route("/repos", repo_routes.ingest_route, methods=["POST"]),
        Route("/repos", repo_routes.list_route, methods=["GET"]),
        Route("/repos/jobs/{job_id}", repo_routes.job_route, methods=["GET"]),
        Route("/repos/{name}", repo_routes.remove_route, methods=["DELETE"]),
        Route("/keys/{label}/authors", access_routes.authors_route, methods=["GET"]),
        Route("/keys/{label}/authors", access_routes.authors_put_route, methods=["PUT"]),
        Route("/namespaces", access_routes.namespaces_create_route, methods=["POST"]),
        Route("/namespaces", access_routes.namespaces_list_route, methods=["GET"]),
        Route("/namespaces/{name}", access_routes.namespaces_delete_route, methods=["DELETE"]),
        Route("/admin/notes", note_routes.old_notes_route, methods=["GET"]),
        Route("/admin/notes/delete", note_routes.delete_route, methods=["POST"]),
        Route("/admin/notes/move", note_routes.move_route, methods=["POST"]),
        Route("/admin/duplicates", note_routes.duplicates_route, methods=["GET"]),
        Route("/admin/consolidate/groups", consolidation_routes.groups_route, methods=["GET"]),
        Route("/admin/consolidate/verdicts", consolidation_routes.verdicts_route, methods=["POST"]),
        Route("/admin/consolidate/undo", consolidation_routes.undo_route, methods=["POST"]),
        Route("/admin/consolidate/actions", consolidation_routes.actions_route, methods=["GET"]),
        Route("/admin/archive", note_routes.archive_route, methods=["POST"]),
        Route("/admin/messages/purge", message_routes.purge_route, methods=["POST"]),
        Route("/admin/restore", note_routes.restore_route, methods=["POST"]),
    ],
)
