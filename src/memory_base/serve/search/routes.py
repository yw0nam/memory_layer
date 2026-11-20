"""REST route for hybrid note search, and the hit serializer it answers with."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import UpstreamUnavailable
from memory_base.retrieval.search import normalize_namespaces
from memory_base.retrieval.search import search
from memory_base.serve.common.http import error
from memory_base.serve.common.http import json_body
from memory_base.serve.common.http import note_scope
from memory_base.serve.decisions.store import repo_scope
from memory_base.serve.notes.store import note_date
from memory_base.serve.search import recall

LOGGER = logging.getLogger(__name__)
# Beyond this a query is a pasted payload, not a question: it costs embedder and BM25
# work no ranking can use.
MAX_QUERY_CHARS = 2000
MAX_BUDGET_TOKENS = 32000
RECALL_MAX_QUERY_CHARS = 4000
RECALL_MAX_TOP_K = 5
RECALL_FIELDS = frozenset({"query", "scope", "notes", "decisions"})
RECALL_DEFAULTS = {"notes": (3, 0.5), "decisions": (2, 0.5)}


def hit_to_dict(hit: Hit) -> dict[str, Any]:
    """Convert a search hit into a JSON-serializable response object."""
    out: dict[str, Any] = {
        "id": hit.meta["id"],
        "kind": hit.meta["kind"],
        "tags": hit.meta["tags"],
        "ref": hit.ref,
        "date": note_date(hit.meta["occurred_at"], hit.ts),
        "score": hit.score,
        "text": hit.text,
    }
    if hit.meta.get("archived"):
        out["archived"] = True
    if hit.meta.get("author"):
        out["author"] = hit.meta["author"]
    if hit.meta.get("shared"):
        out["shared"] = True
    if hit.meta.get("supersedes") is not None:
        out["supersedes"] = hit.meta["supersedes"]
    return out


async def search_route(request: Request) -> JSONResponse:
    """Validate and execute a hybrid search request; `namespaces` narrows it, omitted covers all."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")

    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return error("query must be a non-empty string")
    query = query[:MAX_QUERY_CHARS]
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
    if "since" in body and body["since"] is None:
        return error("since must be an ISO 8601 date or datetime string")
    if "until" in body and body["until"] is None:
        return error("until must be an ISO 8601 date or datetime string")
    if "author" in body and body["author"] is None:
        return error('author must be an author slug or "all"')
    try:
        requested_namespaces = normalize_namespaces(body.get("namespaces"))
        authors, visible_to = note_scope(request.state.author, body.get("author"))
        options: dict[str, Any] = {"include_archived": include_archived}
        if "kind" in body:
            options["kind"] = body["kind"]
        if "tags" in body:
            options["tags"] = body["tags"]
        if "since" in body:
            options["since"] = body["since"]
        if "until" in body:
            options["until"] = body["until"]
        if "min_score" in body:
            options["min_score"] = body["min_score"]
        if authors is not None:
            options["authors"] = authors
        if visible_to is not None:
            options["visible_to"] = visible_to
        if budget_tokens is not None:
            options["budget_tokens"] = budget_tokens
        if requested_namespaces is not None:
            options["namespaces"] = requested_namespaces
        started = time.monotonic()
        hits = await search(query, **options)
        if budget_tokens is None:
            hits = hits[:top_k]
    except UpstreamUnavailable as exc:
        cause = exc.__cause__
        LOGGER.warning(
            "search unavailable: service=%s cause=%s: %s elapsed=%.1fs",
            exc.service,
            type(cause).__name__,
            cause,
            time.monotonic() - started,
        )
        return error(f"search unavailable: {exc}, so memory cannot be attached right now", 503)
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse([hit_to_dict(hit) for hit in hits])


def _lane_options(body: dict[str, Any], lane: str) -> tuple[int, float]:
    """The lane's (top_k, min_score) from the body, else its default."""
    options = body.get(lane, {})
    if not isinstance(options, dict) or not set(options) <= {"top_k", "min_score"}:
        raise ValueError(f"{lane} must be an object with top_k and min_score")
    top_k, min_score = RECALL_DEFAULTS[lane]
    top_k = options.get("top_k", top_k)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= RECALL_MAX_TOP_K:
        raise ValueError(f"{lane}.top_k must be an integer between 1 and {RECALL_MAX_TOP_K}")
    min_score = options.get("min_score", min_score)
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise ValueError(f"{lane}.min_score must be a number between 0 and 1")
    if not 0 <= min_score <= 1:
        raise ValueError(f"{lane}.min_score must be a number between 0 and 1")
    return top_k, min_score


async def recall_route(request: Request) -> JSONResponse:
    """Notes and the caller's repository decisions for one query, scored separately.

    A failing lane is named in `errors` and the other still answers; all failing is a 503.
    """
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return error("query must be a non-empty string")
    if len(query) > RECALL_MAX_QUERY_CHARS:
        return error(f"query must be at most {RECALL_MAX_QUERY_CHARS} chars")
    if unknown := sorted(set(body) - RECALL_FIELDS):
        return error(f"unknown field(s): {', '.join(unknown)}")
    try:
        scope = repo_scope(body["scope"]) if "scope" in body else None
        note_options = _lane_options(body, "notes")
        decision_options = _lane_options(body, "decisions")
    except ValueError as exc:
        return error(str(exc))

    author = request.state.author
    # The note search gets the query's tail: the query ends with the user's prompt.
    lanes = {"notes": recall.recall_notes(query[-MAX_QUERY_CHARS:], author, *note_options)}
    if scope is not None:
        lanes["decisions"] = recall.recall_decisions(query, scope, author, *decision_options)
    started = time.monotonic()
    results = dict(zip(lanes, await asyncio.gather(*lanes.values(), return_exceptions=True)))
    failures = {
        name: result for name, result in results.items() if isinstance(result, BaseException)
    }
    for name, exc in failures.items():
        if isinstance(exc, UpstreamUnavailable):
            LOGGER.warning(
                "recall unavailable: lane=%s service=%s cause=%s: %s elapsed=%.1fs",
                name,
                exc.service,
                type(exc.__cause__).__name__,
                exc.__cause__,
                time.monotonic() - started,
            )
        else:
            LOGGER.error("recall failed: lane=%s", name, exc_info=exc)
    if len(failures) == len(lanes):
        first = next(iter(failures.values()))
        detail = f": {first}" if isinstance(first, UpstreamUnavailable) else ""
        return error(f"recall unavailable{detail}, so memory cannot be attached right now", 503)
    ok = {name: result for name, result in results.items() if name not in failures}
    return JSONResponse(
        {
            "notes": [hit_to_dict(hit) for hit in ok.get("notes", [])],
            "decisions": ok.get("decisions", []),
            "errors": list(failures),
        }
    )
