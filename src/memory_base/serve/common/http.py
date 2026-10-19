"""Shared request/response helpers for serve-layer feature modules."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

TEXT_LIMIT = 2000
AUTHOR_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
RESERVED_AUTHOR = "all"
AUTHOR_BODY_ERROR = "author comes from the X-Author header"
AUTHOR_SCOPE_DESCRIPTION = (
    'Omitted: your own notes and shared notes. "all": every agent\'s notes. '
    "A slug: that agent's notes only."
)


def author_slug(value: Any) -> str | None:
    """The author slug a value names; None when it is missing, malformed, or reserved."""
    if not isinstance(value, str) or value == RESERVED_AUTHOR or not AUTHOR_RE.fullmatch(value):
        return None
    return value


def reject_author(body: Any) -> JSONResponse | None:
    """The 400 a permissive write body carrying `author` gets, else None."""
    return error(AUTHOR_BODY_ERROR) if isinstance(body, dict) and "author" in body else None


async def json_body(request: Request) -> dict[str, Any]:
    """Parse the request body as JSON, requiring a top-level object."""
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


def error(message: str, status: int = 400) -> JSONResponse:
    """Build a `{"error": message}` JSON response with the given status code."""
    return JSONResponse({"error": message}, status_code=status)


def iso(ts: float | None) -> str | None:
    """An epoch timestamp as an ISO 8601 UTC string; None stays None."""
    return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class RequestError(ValueError):
    """A request breaks the schema; REST maps it to 400."""


def object_fields(
    body: Any, required: set[str], optional: set[str] = frozenset()
) -> dict[str, Any]:
    """The body as an object holding every required field and no unknown one."""
    if not isinstance(body, dict):
        raise RequestError("JSON body must be an object")
    unknown = sorted(set(body) - required - optional)
    if unknown:
        raise RequestError(f"unknown field: {', '.join(unknown)}")
    missing = sorted(required - set(body))
    if missing:
        raise RequestError(f"missing field: {', '.join(missing)}")
    return body


def note_scope(request_author: str, requested: str | None) -> tuple[str | None, str | None]:
    """Resolve a note read's author filter into (author_exact, visible_to).

    Omitted means the caller's own notes plus shared ones; "all" covers every author;
    a slug means exactly that author.
    """
    if requested is None:
        return None, request_author
    if requested == RESERVED_AUTHOR:
        return None, None
    if (slug := author_slug(requested)) is None:
        raise RequestError('author must be an author slug or "all"')
    return slug, None


def stripped(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RequestError(f"{name} must be a string")
    return value.strip()


def bounded_text(value: Any, name: str, low: int, high: int) -> str:
    text = stripped(value, name)
    if not low <= len(text) <= high:
        raise RequestError(f"{name} must hold {low} to {high} characters after stripping")
    return text


async def request_body(request: Request) -> dict[str, Any]:
    """The JSON object body; an unreadable one is a RequestError."""
    try:
        return await json_body(request)
    except Exception as exc:
        raise RequestError(f"invalid JSON body: {exc}") from None
