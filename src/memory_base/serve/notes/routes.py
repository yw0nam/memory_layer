"""REST routes for agent notes: saving, listing, and curation, scoped by the caller's key."""

from __future__ import annotations

import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.retrieval.search import normalize_namespaces
from memory_base.serve.common.http import error, json_body
from memory_base.serve.notes import curation, store


def _ids(body: dict[str, Any]) -> list[str] | None:
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
        return None
    return ids


def _admin_scope(key) -> list[str] | None:
    """None means unfiltered (an admin key); otherwise the caller's allowed set."""
    return None if key.is_admin else sorted(key.allowed)


async def save_route(request: Request) -> JSONResponse:
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
    author = body.get("author")
    if not isinstance(author, str) or not author.strip():
        return error("author is required")
    if author not in key.authors:
        return error(f"author {author!r} is not permitted for this key", 403)
    allow_similar = body.get("allow_similar", False)
    if not isinstance(allow_similar, bool):
        return error("allow_similar must be a boolean")
    try:
        result = await store.save_note(
            body.get("content", ""),
            kind=body.get("kind"),
            tags=body.get("tags"),
            supersedes=body.get("supersedes"),
            namespace=namespace,
            occurred_at=body.get("occurred_at"),
            author=author,
            allow_similar=allow_similar,
        )
    except store.SimilarNotesError as exc:
        return JSONResponse({"error": str(exc), "similar": exc.similar}, status_code=409)
    except store.CredentialNoteError as exc:
        return error(str(exc), 409)
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(result)


async def list_route(request: Request) -> JSONResponse:
    """List agent notes by filters alone — no query, no embedding — scoped like search."""
    key = request.state.key
    params = request.query_params
    try:
        limit = int(params.get("limit", str(store.LIST_NOTES_DEFAULT_LIMIT)))
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
        rows = await store.list_notes(
            tags=params.getlist("tags") or None,
            kind=params.get("kind") or None,
            namespaces=scope,
            include_archived=include_archived == "true",
            since=params.get("since"),
            until=params.get("until"),
            author=params.get("author") or None,
            limit=limit,
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(rows)


async def old_route(request: Request) -> JSONResponse:
    """List active agent notes older than a requested age, scoped to the caller's namespaces."""
    try:
        older_than_days = int(request.query_params.get("older_than_days", "90"))
    except ValueError:
        return error("older_than_days must be an integer")
    rows = await curation.list_old_notes(
        older_than_days, namespaces=_admin_scope(request.state.key)
    )
    return JSONResponse(rows)


async def delete_route(request: Request) -> JSONResponse:
    """Preview or delete selected agent notes, scoped to the caller's namespaces."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    ids = _ids(body)
    if ids is None:
        return error("ids must be a non-empty list")
    scope = _admin_scope(request.state.key)
    rows = await curation.notes_by_ids(ids, namespaces=scope)
    if {row["id"] for row in rows} != set(ids):
        return error("ids must refer only to agent_note rows")
    if body.get("confirm") is True:
        deleted = await curation.delete_notes(ids, namespaces=scope)
        return JSONResponse({"deleted": deleted})
    return JSONResponse({"rows": rows})


async def move_route(request: Request) -> JSONResponse:
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
        result = await curation.move_notes(ids, target_namespace)
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(result)


async def duplicates_route(request: Request) -> JSONResponse:
    """List active near-duplicate agent-note pairs, scoped to the caller's namespaces."""
    try:
        threshold = float(request.query_params.get("threshold", "0.9"))
    except ValueError:
        return error("threshold must be a number")
    kind = request.query_params.get("kind") or None
    if kind is not None and kind not in store.NOTE_KINDS:
        return error(f"kind must be one of {store.NOTE_KINDS}")
    try:
        limit = int(request.query_params.get("limit", "50"))
    except ValueError:
        return error("limit must be an integer")
    pairs = await curation.find_duplicates(
        threshold, kind, limit, namespaces=_admin_scope(request.state.key)
    )
    return JSONResponse({"pairs": pairs})


async def archive_route(request: Request) -> JSONResponse:
    """Preview or archive the agent notes named by ids, or the cold ones, in scope."""
    key = request.state.key
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    ids = None
    if "ids" in body:
        ids = _ids(body)
        if ids is None:
            return error("ids must be a non-empty list")
    author = body.get("author")
    if ids is not None and author is None:
        return error("author is required")
    if author is not None:
        if not isinstance(author, str) or not author.strip():
            return error("author must be a non-empty string")
        if author not in key.authors:
            return error(f"author {author!r} is not permitted for this key", 403)
    now = time.time()
    scope = _admin_scope(key)
    if ids is not None:
        rows = await curation.rows_by_ids(ids, namespaces=scope, notes_only=True)
        if {row["id"] for row in rows} != set(ids):
            return error("ids must refer only to rows in the caller's scope")
        if body.get("confirm") is True:
            archived = await curation.archive_rows(ids, now, namespaces=scope, archived_by=author)
            return JSONResponse({"archived": archived})
        return JSONResponse({"notes_to_archive": rows})
    candidates = await curation.archive_candidates(now, namespaces=scope)
    if body.get("confirm") is True:
        archived = await curation.archive_rows(
            [row["id"] for row in candidates], now, namespaces=scope, archived_by=author
        )
        return JSONResponse({"archived": archived})
    return JSONResponse({"notes_to_archive": candidates})


async def restore_route(request: Request) -> JSONResponse:
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
        restored = await curation.restore_rows(ids, namespaces=scope)
        return JSONResponse({"restored": restored})
    rows = await curation.rows_by_ids(ids, namespaces=scope)
    return JSONResponse({"rows": rows})
