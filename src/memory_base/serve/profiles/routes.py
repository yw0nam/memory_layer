"""REST routes for agent-owned profiles: request validation, authority, and status codes.

Authority comes only from the key's authors: an owner reads and writes as itself; a key
carrying the `user` author reads every owner. The user part changes with the user's consent,
recorded on the version; the `import` method needs an admin key carrying the `user` author.
"""

from __future__ import annotations

import re
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.core.secrets import find_secret
from memory_base.serve.common import consent
from memory_base.serve.common.consent import CONSENT_METHODS, USER_AUTHOR
from memory_base.serve.common.http import (
    RequestError,
    bounded_text,
    error,
    object_fields,
    request_body,
    stripped,
)
from memory_base.serve.access.keys import AUTHOR_RE
from memory_base.serve.profiles import store

PARTS = ("self", "user")
RESERVED_OWNERS = frozenset({USER_AUTHOR, "consolidator"})
DEFAULT_SELF_MAX_CHARS = 4000
DEFAULT_USER_MAX_CHARS = 3000
MIN_MAX_CHARS = 200
MAX_MAX_CHARS = 20000
MAX_REASON_CHARS = 1000
MAX_BASE_VERSION = 2_147_483_647
DEFAULT_LIMIT = 20
MAX_LIMIT = 200

_DIGITS = re.compile(r"[0-9]+")


class Refused(ValueError):
    """A text fails a content check; nothing is stored. REST maps it to 400."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details


# ---- request validation -----------------------------------------------------


def _owner(value: Any) -> str:
    if not isinstance(value, str) or not AUTHOR_RE.fullmatch(value) or value in RESERVED_OWNERS:
        raise RequestError(
            "owner must be an author slug matching ^[a-z0-9][a-z0-9-]{0,39}$ "
            "other than 'user' or 'consolidator'"
        )
    return value


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise RequestError(f"{name} must be an integer between {low} and {high}")
    return value


def _no_secret(text: str, name: str) -> None:
    secret = find_secret(text)
    if secret is not None:
        raise Refused(f"{name} contains a credential ({secret})")


def _content(value: Any, max_chars: int) -> str:
    content = stripped(value, "content")
    if len(content) > max_chars:
        raise Refused(
            f"content is {len(content)} characters, over max_chars {max_chars}",
            chars=len(content),
            max_chars=max_chars,
        )
    _no_secret(content, "content")
    return content


def _bounded_text(value: Any, name: str, low: int, high: int) -> str:
    text = bounded_text(value, name, low, high)
    _no_secret(text, name)
    return text


def _max_chars(body: dict[str, Any], default: int) -> int:
    return _integer(body.get("max_chars", default), "max_chars", MIN_MAX_CHARS, MAX_MAX_CHARS)


def parse_self(body: Any) -> tuple[str, str]:
    """`{owner, content, max_chars?}` → (owner, stripped content)."""
    fields = object_fields(body, {"owner", "content"}, {"max_chars"})
    owner = _owner(fields["owner"])
    return owner, _content(fields["content"], _max_chars(fields, DEFAULT_SELF_MAX_CHARS))


def parse_user(body: Any) -> tuple[str, str, str, int, str, bool]:
    """`{owner, content, reason, base_version, consent_method, dry_run?, max_chars?}` →
    (owner, content, reason, base, method, dry_run)."""
    fields = object_fields(
        body,
        {"owner", "content", "reason", "base_version", "consent_method"},
        {"dry_run", "max_chars"},
    )
    owner = _owner(fields["owner"])
    base_version = _integer(fields["base_version"], "base_version", 0, MAX_BASE_VERSION)
    content = _content(fields["content"], _max_chars(fields, DEFAULT_USER_MAX_CHARS))
    reason = _bounded_text(fields["reason"], "reason", 1, MAX_REASON_CHARS)
    method = fields["consent_method"]
    if method not in CONSENT_METHODS:
        raise RequestError(f"consent_method must be one of {CONSENT_METHODS}")
    dry_run = fields.get("dry_run", False)
    if not isinstance(dry_run, bool):
        raise RequestError("dry_run must be a boolean")
    return owner, content, reason, base_version, method, dry_run


def _query(request: Request, allowed: tuple[str, ...]) -> dict[str, str]:
    """The query parameters, each known and given at most once."""
    names = [name for name, _ in request.query_params.multi_items()]
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise RequestError(f"unknown query parameter: {', '.join(unknown)}")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise RequestError(f"query parameter given more than once: {', '.join(repeated)}")
    return dict(request.query_params)


def _limit(value: str | None) -> int:
    if value is None:
        return DEFAULT_LIMIT
    if not _DIGITS.fullmatch(value) or not 1 <= int(value) <= MAX_LIMIT:
        raise RequestError(f"limit must be an integer between 1 and {MAX_LIMIT}")
    return int(value)


# ---- routes -----------------------------------------------------------------


def _may_read(key: Any, owner: str) -> bool:
    return owner in key.authors or USER_AUTHOR in key.authors


def _owner_denied(owner: str) -> JSONResponse:
    return error(f"owner {owner!r} is not one of this key's authors", 403)


def _refused(exc: Refused) -> JSONResponse:
    return JSONResponse({"error": str(exc), **exc.details}, status_code=400)


def _stale(exc: store.Stale) -> JSONResponse:
    return JSONResponse({"error": "stale", "version": exc.version}, status_code=409)


async def profile_route(request: Request) -> JSONResponse:
    """One owner's served parts and their versions."""
    key = request.state.key
    try:
        owner = _owner(_query(request, ("owner",)).get("owner"))
    except RequestError as exc:
        return error(str(exc))
    if not _may_read(key, owner):
        return _owner_denied(owner)
    return JSONResponse(await store.read(owner))


async def self_route(request: Request) -> JSONResponse:
    """Replace the owner's `self` part; `unchanged` when it equals the latest version."""
    key = request.state.key
    try:
        _query(request, ())
        owner, content = parse_self(await request_body(request))
    except RequestError as exc:
        return error(str(exc))
    except Refused as exc:
        return _refused(exc)
    if owner not in key.authors:
        return _owner_denied(owner)
    return JSONResponse(await store.write_self(owner, content))


async def user_route(request: Request) -> JSONResponse:
    """Replace the owner's `user` part with the user's consent; 409 when its base is stale.

    With `dry_run`, every check runs, nothing is written, and the current content returns."""
    key = request.state.key
    try:
        _query(request, ())
        owner, content, reason, base_version, method, dry_run = parse_user(
            await request_body(request)
        )
    except RequestError as exc:
        return error(str(exc))
    except Refused as exc:
        return _refused(exc)
    if owner not in key.authors:
        return _owner_denied(owner)
    if method == "import" and not consent.may_import(key):
        return error("consent_method import needs an admin key carrying the 'user' author", 403)
    try:
        result = await store.write_user(owner, content, reason, base_version, method, dry_run)
    except store.Stale as exc:
        return _stale(exc)
    return JSONResponse(result)


async def versions_route(request: Request) -> JSONResponse:
    """One part's versions newest first."""
    key = request.state.key
    try:
        query = _query(request, ("owner", "part", "limit"))
        owner = _owner(query.get("owner"))
        part = query.get("part")
        if part not in PARTS:
            raise RequestError(f"part must be one of {PARTS}")
        limit = _limit(query.get("limit"))
    except RequestError as exc:
        return error(str(exc))
    if not _may_read(key, owner):
        return _owner_denied(owner)
    return JSONResponse(await store.versions(owner, part, limit))
