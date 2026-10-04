"""Starlette REST API for memory search and storage."""

from __future__ import annotations

import asyncio
import logging
import os
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
from memory_base.core.llm import resolve_llm_provider
from memory_base.core.logger import setup_logging
from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import UpstreamUnavailable
from memory_base.retrieval.search import normalize_namespaces
from memory_base.retrieval.search import search
from memory_base.serve import access_log
from memory_base.serve import ingest_api
from memory_base.serve import job_store
from memory_base.serve import keys
from memory_base.serve import namespaces
from memory_base.serve import repos
from memory_base.serve import tables
from memory_base.serve.auth import ApiKeyAuthMiddleware
from memory_base.serve.common.http import TEXT_LIMIT
from memory_base.serve.common.http import error
from memory_base.serve.common.http import json_body
from memory_base.serve.consolidation import routes as consolidation_routes
from memory_base.serve.messages import routes as message_routes
from memory_base.serve.notes import routes as note_routes
from memory_base.serve.notes.store import note_date
from memory_base.serve.profiles import routes as profile_routes

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


async def keys_authors_route(request: Request) -> JSONResponse:
    """Report a label's author allowlist; a non-admin key may read only its own label."""
    key = request.state.key
    label = request.path_params["label"]
    if not (key.is_admin or key.label == label):
        return error("not permitted to read another label's authors", 403)
    authors = await keys.get_authors(label)
    if authors is None:
        return JSONResponse({"error": f"unknown key label: {label}"}, status_code=404)
    return JSONResponse({"label": label, "authors": authors})


async def keys_authors_put_route(request: Request) -> JSONResponse:
    """Replace a label's author allowlist; admin keys only."""
    if not request.state.key.is_admin:
        return error("admin key required", 403)
    label = request.path_params["label"]
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    try:
        authors = keys.validate_authors(body.get("authors"))
    except keys.AuthorError as exc:
        return error(str(exc))
    stored = await keys.set_authors(label, authors)
    if stored is None:
        return JSONResponse({"error": f"unknown key label: {label}"}, status_code=404)
    return JSONResponse({"label": label, "authors": stored})


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
        Route("/keys/{label}/authors", keys_authors_route, methods=["GET"]),
        Route("/keys/{label}/authors", keys_authors_put_route, methods=["PUT"]),
        Route("/namespaces", namespaces_create_route, methods=["POST"]),
        Route("/namespaces", namespaces_list_route, methods=["GET"]),
        Route("/namespaces/{name}", namespaces_delete_route, methods=["DELETE"]),
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
