"""Shared request/response helpers for serve-layer feature modules."""

from __future__ import annotations

import re
from typing import Any, Literal

from starlette.requests import Request
from starlette.responses import JSONResponse

TEXT_LIMIT = 2000
AUTHOR_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
RESERVED_AUTHOR = "all"
AUTHOR_BODY_ERROR = "author comes from the X-Author header"
AUTHOR_SCOPE_DESCRIPTION = (
    "Omitted: your own notes and shared notes. Another agent's notes come back only "
    "when you name it: a list of slugs returns exactly those agents' notes, e.g. "
    '["natsume", "claude-code"]; include your own slug to keep yours. '
    '"all": every agent\'s notes.'
)
AuthorScope = list[str] | Literal["all"] | None


def author_params(author: AuthorScope) -> list[tuple[str, str]]:
    """The repeated `author` query params a tool sends for its author scope."""
    if author is None:
        return []
    return [("author", name) for name in ([author] if isinstance(author, str) else author)]


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


class RequestError(ValueError):
    """A request breaks the schema; REST maps it to 400."""


def note_scope(
    request_author: str, requested: str | list[str] | None
) -> tuple[list[str] | None, str | None]:
    """Resolve a note read's author filter into (authors, visible_to).

    Omitted means the caller's own notes plus shared ones; "all" covers every author;
    a slug or a list of slugs means exactly those authors.
    """
    if requested is None:
        return None, request_author
    if requested in (RESERVED_AUTHOR, [RESERVED_AUTHOR]):
        return None, None
    names = [requested] if isinstance(requested, str) else requested
    slugs = [author_slug(name) for name in names] if isinstance(names, list) else []
    if not slugs or None in slugs:
        raise RequestError('author must be "all" or a list of author slugs')
    return slugs, None
