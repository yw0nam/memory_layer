"""REST routes for the verbatim turn lane: request validation and status codes."""

from __future__ import annotations

import uuid
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.retrieval.search import normalize_namespaces, normalize_time_range
from memory_base.serve.access.namespaces import DEFAULT_NAMESPACE
from memory_base.serve.common.http import (
    RequestError,
    bounded_text,
    error,
    object_fields,
    request_body,
)
from memory_base.serve.common.origins import ORIGINS
from memory_base.serve.turns import store

MAX_SESSION_ID_CHARS = 200
MAX_REPO_CHARS = 300
REPO_PREFIX = "repo:"


def _choice(value: Any, name: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise RequestError(f"{name} must be one of {allowed}")
    return value


def _repo(value: Any) -> str | None:
    if value is None:
        return None
    repo = bounded_text(value, "repo", 1, MAX_REPO_CHARS)
    if not repo.startswith(REPO_PREFIX):
        raise RequestError(f"repo must start with {REPO_PREFIX!r}")
    return repo


def parse_turn(body: Any) -> dict[str, Any]:
    """The request body as a validated turn row for the store."""
    fields = object_fields(
        body,
        {"id", "origin", "session_id", "role", "content"},
        {"reasoning", "repo", "namespace"},
    )
    try:
        turn_id = str(uuid.UUID(str(fields["id"])))
    except ValueError:
        raise RequestError("id must be a UUID") from None
    namespace = fields.get("namespace", DEFAULT_NAMESPACE)
    if not isinstance(namespace, str) or not namespace.strip():
        raise RequestError("namespace must be a non-empty string")
    reasoning = fields.get("reasoning")
    return {
        "id": turn_id,
        "namespace": namespace,
        "origin": _choice(fields["origin"], "origin", ORIGINS),
        "session_id": bounded_text(fields["session_id"], "session_id", 1, MAX_SESSION_ID_CHARS),
        "role": _choice(fields["role"], "role", store.ROLES),
        "content": bounded_text(fields["content"], "content", 1, store.MAX_TEXT_CHARS),
        "reasoning": None
        if reasoning is None
        else bounded_text(reasoning, "reasoning", 1, store.MAX_TEXT_CHARS),
        "repo": _repo(fields.get("repo")),
    }


async def append_route(request: Request) -> JSONResponse:
    """Append one turn to its session; a replay of the same payload returns the stored seq."""
    try:
        return JSONResponse(await store.append(parse_turn(await request_body(request))))
    except store.Conflict as exc:
        return error(str(exc), 409)
    except ValueError as exc:
        return error(str(exc))


def _int(request: Request, name: str, default: int, low: int, high: int) -> int:
    raw = request.query_params.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _filters(request: Request) -> dict[str, Any]:
    params = request.query_params
    since, until = normalize_time_range(params.get("since"), params.get("until"))
    origin = params.get("origin") or None
    if origin is not None:
        _choice(origin, "origin", ORIGINS)
    return {
        "namespaces": normalize_namespaces(params.getlist("namespace") or None),
        "since": since,
        "until": until,
        "origin": origin,
        "repo": params.get("repo") or None,
        "limit": _int(request, "limit", store.DEFAULT_LIMIT, 1, store.MAX_LIMIT),
    }


async def sessions_route(request: Request) -> JSONResponse:
    """List sessions newest first by their last turn."""
    try:
        return JSONResponse(await store.list_sessions(**_filters(request)))
    except ValueError as exc:
        return error(str(exc))


async def session_route(request: Request) -> JSONResponse:
    """Read one session's turns, or the matches of `contains` with their neighbours."""
    params = request.query_params
    try:
        origin = _choice(request.path_params["origin"], "origin", ORIGINS)
        return JSONResponse(
            await store.read_session(
                normalize_namespaces(params.getlist("namespace") or None),
                origin,
                request.path_params["session_id"],
                params.get("contains") or None,
                _int(request, "offset", 1, 1, 2**31 - 1),
                _int(request, "limit", store.DEFAULT_LIMIT, 1, store.MAX_LIMIT),
            )
        )
    except store.NotFound as exc:
        return error(str(exc), 404)
    except ValueError as exc:
        return error(str(exc))


async def search_route(request: Request) -> JSONResponse:
    """Find turns whose content holds a literal substring, newest first."""
    try:
        filters = _filters(request)
        return JSONResponse(await store.search(q=request.query_params.get("q", ""), **filters))
    except ValueError as exc:
        return error(str(exc))


async def delete_session_route(request: Request) -> JSONResponse:
    """Hard-delete a session's turns."""
    try:
        origin = _choice(request.path_params["origin"], "origin", ORIGINS)
    except RequestError as exc:
        return error(str(exc))
    deleted = await store.delete_session(origin, request.path_params["session_id"])
    return JSONResponse({"deleted": deleted})
