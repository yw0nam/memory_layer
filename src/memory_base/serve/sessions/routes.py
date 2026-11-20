"""REST routes for the session record: request validation and status codes."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.retrieval.search import normalize_time_range
from memory_base.serve.common.http import RequestError, error, reject_author
from memory_base.serve.common.scope import normalize_scope
from memory_base.serve.sessions import store

MAX_SESSION_ID_CHARS = 200
MAX_HOST_CHARS = 255
MAX_CWD_CHARS = 4096
MAX_NAME_CHARS = 200
_REQUIRED = {"id", "session_id", "role", "content", "event_at", "host", "cwd"}
_OPTIONAL = {"tool_name", "tool_use_id", "repo"}


def _text(value: Any, name: str, low: int, high: int) -> str:
    """The string as sent (never stripped), without NUL characters Postgres cannot store."""
    if not isinstance(value, str):
        raise RequestError(f"{name} must be a string")
    value = value.replace("\x00", "")
    if not low <= len(value) <= high:
        raise RequestError(f"{name} must hold {low} to {high} characters")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name, 1, MAX_NAME_CHARS)


def _repo(value: Any) -> str | None:
    """The normalized `repo:` scope, or None when absent."""
    if value is None:
        return None
    try:
        repo = normalize_scope(value)
    except ValueError as exc:
        raise RequestError(f"repo: {exc}") from None
    if not repo.startswith("repo:"):
        raise RequestError("repo must be repo:<host>/<path>")
    return repo


def parse_event(raw: Any) -> dict[str, Any]:
    """One request event as a validated row for the store."""
    if not isinstance(raw, dict):
        raise RequestError("each event must be an object")
    if unknown := sorted(set(raw) - _REQUIRED - _OPTIONAL):
        raise RequestError(f"unknown event field: {', '.join(unknown)}")
    if missing := sorted(_REQUIRED - set(raw)):
        raise RequestError(f"missing event field: {', '.join(missing)}")
    try:
        event_id = str(uuid.UUID(str(raw["id"])))
    except ValueError:
        raise RequestError("id must be a UUID") from None
    if raw["role"] not in store.ROLES:
        raise RequestError(f"role must be one of {store.ROLES}")
    try:
        event_at = datetime.fromisoformat(_text(raw["event_at"], "event_at", 1, 64))
    except ValueError:
        raise RequestError("event_at must be an ISO 8601 timestamp") from None
    return {
        "id": event_id,
        "session_id": _text(raw["session_id"], "session_id", 1, MAX_SESSION_ID_CHARS),
        "role": raw["role"],
        "content": _text(raw["content"], "content", 0, store.MAX_CONTENT_CHARS),
        "tool_name": _optional_text(raw.get("tool_name"), "tool_name"),
        "tool_use_id": _optional_text(raw.get("tool_use_id"), "tool_use_id"),
        "event_at": event_at if event_at.tzinfo else event_at.replace(tzinfo=timezone.utc),
        "repo": _repo(raw.get("repo")),
        "host": _text(raw["host"], "host", 1, MAX_HOST_CHARS),
        "cwd": _text(raw["cwd"], "cwd", 1, MAX_CWD_CHARS),
    }


async def write_route(request: Request) -> JSONResponse:
    """Store a batch of events; a replay of the same payload counts as known."""
    raw = await request.body()
    if len(raw) > store.MAX_BODY_BYTES:
        return error(f"request body must be at most {store.MAX_BODY_BYTES} bytes")
    try:
        body = json.loads(raw)
    except ValueError as exc:
        return error(f"invalid JSON body: {exc}")
    if rejected := reject_author(body):
        return rejected
    try:
        if not isinstance(body, dict) or set(body) != {"events"}:
            raise RequestError('the body must be an object with one field, "events"')
        events = body["events"]
        if not isinstance(events, list) or not 1 <= len(events) <= store.MAX_EVENTS:
            raise RequestError(f"events must be a list of 1 to {store.MAX_EVENTS} events")
        rows = [parse_event(event) for event in events]
    except RequestError as exc:
        return error(str(exc))
    return JSONResponse(await store.store_events(request.state.author, rows))


def _int(request: Request, name: str, default: int, low: int, high: int) -> int:
    raw = request.query_params.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError:
        raise RequestError(f"{name} must be an integer") from None
    if not low <= value <= high:
        raise RequestError(f"{name} must be between {low} and {high}")
    return value


def _role(request: Request) -> str | None:
    role = request.query_params.get("role") or None
    if role is not None and role not in store.ROLES:
        raise RequestError(f"role must be one of {store.ROLES}")
    return role


def _contains(request: Request, name: str) -> str | None:
    value = request.query_params.get(name) or None
    if value is not None and len(value) < store.MIN_QUERY_CHARS:
        raise RequestError(f"{name} must hold at least {store.MIN_QUERY_CHARS} characters")
    return value


async def list_route(request: Request) -> JSONResponse:
    """List the caller's sessions newest first."""
    params = request.query_params
    try:
        since, until = normalize_time_range(params.get("since"), params.get("until"))
        rows = await store.list_sessions(
            author=request.state.author,
            repo=_repo(params.get("repo") or None),
            since=since,
            until=until,
            limit=_int(request, "limit", 20, 1, store.LIST_MAX_LIMIT),
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(rows)


async def read_route(request: Request) -> JSONResponse:
    """Read one of the caller's sessions, or the matches of `contains` with their neighbours."""
    try:
        return JSONResponse(
            await store.read_session(
                author=request.state.author,
                session_id=request.path_params["session_id"],
                from_seq=_int(request, "from_seq", 1, 1, 2**31 - 1),
                limit=_int(request, "limit", 50, 1, store.READ_MAX_LIMIT),
                role=_role(request),
                contains=_contains(request, "contains"),
            )
        )
    except store.NotFound as exc:
        return error(str(exc), 404)
    except ValueError as exc:
        return error(str(exc))


async def search_route(request: Request) -> JSONResponse:
    """Find the caller's events whose content holds a literal substring, newest first."""
    try:
        q = _contains(request, "q")
        if q is None:
            raise RequestError(f"q must hold at least {store.MIN_QUERY_CHARS} characters")
        return JSONResponse(
            await store.search(
                author=request.state.author,
                q=q,
                repo=_repo(request.query_params.get("repo") or None),
                role=_role(request),
                limit=_int(request, "limit", 20, 1, store.SEARCH_MAX_LIMIT),
            )
        )
    except ValueError as exc:
        return error(str(exc))


async def delete_route(request: Request) -> JSONResponse:
    """Hard-delete the caller's rows of a session."""
    deleted = await store.delete_session(
        author=request.state.author, session_id=request.path_params["session_id"]
    )
    return JSONResponse({"deleted": deleted})
