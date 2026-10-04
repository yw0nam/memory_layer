"""REST route for hybrid search over memory and code, and the hit serializer it answers with."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import UpstreamUnavailable
from memory_base.retrieval.search import normalize_namespaces
from memory_base.retrieval.search import search
from memory_base.serve.common.http import TEXT_LIMIT
from memory_base.serve.common.http import error
from memory_base.serve.common.http import json_body
from memory_base.serve.notes.store import note_date
from memory_base.serve.search import access_log

SOURCES = ("all", "code", "memory")
# Beyond this a query is a pasted payload, not a question: it costs embedder and BM25
# work no ranking can use.
MAX_QUERY_CHARS = 2000
MAX_BUDGET_TOKENS = 32000


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
