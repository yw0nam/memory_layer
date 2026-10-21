"""REST routes for agent notes: saving, listing, and curation."""

from __future__ import annotations

import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.retrieval.search import normalize_namespaces
from memory_base.serve.access.namespaces import DEFAULT_NAMESPACE
from memory_base.serve.common.http import (
    RequestError,
    error,
    json_body,
    note_scope,
    object_fields,
    reject_author,
)
from memory_base.serve.common.origins import ORIGINS
from memory_base.serve.notes import curation, store


def _ids(body: dict[str, Any]) -> list[str] | None:
    ids = body.get("ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
        return None
    return ids


def _missing(ids: list[str], rows: list[dict], noun: str) -> JSONResponse | None:
    """A 400 naming the requested ids absent from rows; None when every id was found."""
    missing = sorted(set(ids) - {row["id"] for row in rows})
    if not missing:
        return None
    return JSONResponse(
        {
            "error": f"{len(missing)} of {len(set(ids))} ids are not {noun}: {', '.join(missing)}",
            "missing": missing,
        },
        status_code=400,
    )


def _source_session(body: dict[str, Any]) -> dict[str, str] | None:
    """The request's source_session, validated; None when absent."""
    source = body.get("source_session")
    if source is None:
        return None
    if (
        not isinstance(source, dict)
        or set(source) != {"origin", "session_id"}
        or source["origin"] not in ORIGINS
        or not isinstance(source["session_id"], str)
        or not source["session_id"].strip()
    ):
        raise ValueError(f"source_session must hold origin (one of {ORIGINS}) and session_id")
    return source


async def save_route(request: Request) -> JSONResponse:
    """Validate and store an agent-authored memory request; omitted namespace lands in `default`."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    if rejected := reject_author(body):
        return rejected

    namespace = body.get("namespace", DEFAULT_NAMESPACE)
    if not isinstance(namespace, str) or not namespace.strip():
        return error("namespace must be a non-empty string")
    if "occurred_at" in body and body["occurred_at"] is None:
        return error("occurred_at must be an ISO 8601 date or datetime string")
    try:
        source_session = _source_session(body)
    except ValueError as exc:
        return error(str(exc))
    allow_similar = body.get("allow_similar", False)
    if not isinstance(allow_similar, bool):
        return error("allow_similar must be a boolean")
    shared = body.get("shared", False)
    if not isinstance(shared, bool):
        return error("shared must be a boolean")
    try:
        result = await store.save_note(
            body.get("content", ""),
            kind=body.get("kind"),
            tags=body.get("tags"),
            entities=body.get("entities"),
            supersedes=body.get("supersedes"),
            namespace=namespace,
            occurred_at=body.get("occurred_at"),
            author=request.state.author,
            allow_similar=allow_similar,
            shared=shared,
            source_session=source_session,
        )
    except store.SimilarNotesError as exc:
        return JSONResponse({"error": str(exc), "similar": exc.similar}, status_code=409)
    except store.CredentialNoteError as exc:
        return error(str(exc), 409)
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(result)


async def list_route(request: Request) -> JSONResponse:
    """List agent notes by filters alone — no query, no embedding."""
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
        authors, visible_to = note_scope(request.state.author, params.getlist("author") or None)
    except ValueError as exc:
        return error(str(exc))
    try:
        rows = await store.list_notes(
            tags=params.getlist("tags") or None,
            entity=params.get("entity") or None,
            kind=params.get("kind") or None,
            namespaces=requested_namespaces,
            include_archived=include_archived == "true",
            since=params.get("since"),
            until=params.get("until"),
            authors=authors,
            visible_to=visible_to,
            limit=limit,
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(rows)


async def old_notes_route(request: Request) -> JSONResponse:
    """List active agent notes older than a requested age."""
    try:
        older_than_days = int(request.query_params.get("older_than_days", "90"))
    except ValueError:
        return error("older_than_days must be an integer")
    try:
        authors, visible_to = note_scope(
            request.state.author, request.query_params.getlist("author") or None
        )
    except ValueError as exc:
        return error(str(exc))
    rows = await curation.list_old_notes(older_than_days, authors=authors, visible_to=visible_to)
    return JSONResponse(rows)


async def delete_route(request: Request) -> JSONResponse:
    """Preview or delete selected agent notes."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    if rejected := reject_author(body):
        return rejected
    ids = _ids(body)
    if ids is None:
        return error("ids must be a non-empty list")
    rows = await curation.notes_by_ids(ids)
    if refused := _missing(ids, rows, "agent notes"):
        return refused
    if body.get("confirm") is True:
        deleted = await curation.delete_notes(ids)
        return JSONResponse({"deleted": deleted})
    return JSONResponse({"rows": rows})


async def move_route(request: Request) -> JSONResponse:
    """Move agent notes into another registered namespace."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    if rejected := reject_author(body):
        return rejected
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
    """List active near-duplicate agent-note pairs."""
    try:
        threshold = float(request.query_params.get("threshold", "0.75"))
    except ValueError:
        return error("threshold must be a number")
    kind = request.query_params.get("kind") or None
    if kind is not None and kind not in store.NOTE_KINDS:
        return error(f"kind must be one of {store.NOTE_KINDS}")
    try:
        limit = int(request.query_params.get("limit", "50"))
    except ValueError:
        return error("limit must be an integer")
    try:
        authors, visible_to = note_scope(
            request.state.author, request.query_params.getlist("author") or None
        )
    except ValueError as exc:
        return error(str(exc))
    pairs = await curation.find_duplicates(
        threshold, kind, limit, authors=authors, visible_to=visible_to
    )
    return JSONResponse({"pairs": pairs})


async def archive_route(request: Request) -> JSONResponse:
    """Preview or archive the agent notes named by ids, or the cold ones in scope."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    if rejected := reject_author(body):
        return rejected
    ids = None
    if "ids" in body:
        ids = _ids(body)
        if ids is None:
            return error("ids must be a non-empty list")
    now = time.time()
    if ids is not None:
        rows = await curation.rows_by_ids(ids, notes_only=True)
        if refused := _missing(ids, rows, "active notes"):
            return refused
        if body.get("confirm") is True:
            archived = await curation.archive_rows(ids, now, archived_by=request.state.author)
            return JSONResponse({"archived": archived})
        return JSONResponse({"notes_to_archive": rows})
    try:
        authors, visible_to = note_scope(
            request.state.author, request.query_params.getlist("author") or None
        )
    except ValueError as exc:
        return error(str(exc))
    candidates = await curation.archive_candidates(now, authors=authors, visible_to=visible_to)
    if body.get("confirm") is True:
        archived = await curation.archive_rows(
            [row["id"] for row in candidates], now, archived_by=request.state.author
        )
        return JSONResponse({"archived": archived})
    return JSONResponse({"notes_to_archive": candidates})


async def restore_route(request: Request) -> JSONResponse:
    """Preview or restore selected memory rows."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    if rejected := reject_author(body):
        return rejected
    ids = _ids(body)
    if ids is None:
        return error("ids must be a non-empty list")
    rows = await curation.rows_by_ids(ids)
    if refused := _missing(ids, rows, "stored rows"):
        return refused
    if body.get("confirm") is True:
        restored = await curation.restore_rows(ids)
        return JSONResponse({"restored": restored})
    return JSONResponse({"rows": rows})


MAX_LINK_IDS = 200


async def link_entities_route(request: Request) -> JSONResponse:
    """Preview or apply one add/remove delta of entity slugs to existing notes."""
    try:
        body = object_fields(await json_body(request), {"ids"}, {"add", "remove", "confirm"})
        ids = _ids(body)
        if ids is None or len(ids) > MAX_LINK_IDS or len(set(ids)) != len(ids):
            raise RequestError(f"ids must hold 1 to {MAX_LINK_IDS} unique note ids")
        confirm = body.get("confirm", False)
        if not isinstance(confirm, bool):
            raise RequestError("confirm must be a boolean")
        add = store.normalize_entities(body.get("add"))
        remove = store.normalize_entities(body.get("remove"))
        if not add and not remove:
            raise RequestError("add or remove must name at least one slug")
        if both := sorted(set(add) & set(remove)):
            raise RequestError(f"slugs in both add and remove: {', '.join(both)}")
    except (RequestError, ValueError) as exc:
        return error(str(exc))
    try:
        result = await curation.link_entities(ids, add, remove, request.state.author, confirm)
    except curation.LinkRefused as exc:
        return JSONResponse({"error": str(exc), "ids": exc.ids}, status_code=400)
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(result)
