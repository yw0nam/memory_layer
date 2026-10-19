"""The two request headers: the API key and the author slug.

Every REST request (except /health*) must carry a valid ``X-API-Key`` header and an
``X-Author`` header holding an author slug. ApiKeyAuthMiddleware looks the key up (by
sha256 hash), attaches a KeyIdentity to ``request.state.key``, and puts the author on
``request.state.author``. A key grants nothing beyond passing this gate; the author
attributes writes and is the default scope of note reads. Fail-closed: no key, an
unknown key, or a revoked key all get 401; a missing or malformed author gets 400.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once

API_KEY_HEADER = "x-api-key"
AUTHOR_HEADER = "x-author"
AUTHOR_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
RESERVED_AUTHOR = "all"
EXEMPT_PATH_PREFIX = "/health"


def hash_key(plaintext: str) -> str:
    """The sha256 hex digest stored in api_keys.key_hash; plaintext is never stored."""
    return hashlib.sha256(plaintext.encode()).hexdigest()


def generate_key() -> str:
    """A fresh random plaintext API key."""
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class KeyIdentity:
    """A request's authenticated caller: the key's hash and the label of its client."""

    key_id: str
    label: str


async def authenticate_request(plaintext_key: str) -> KeyIdentity | None:
    """Look up a plaintext key; None for missing/unknown/revoked."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        key_id = hash_key(plaintext_key)
        label = await conn.fetchval(
            f'SELECT label FROM "{PG_SCHEMA}".api_keys WHERE key_hash = $1 AND revoked_at IS NULL',
            key_id,
        )
        return None if label is None else KeyIdentity(key_id=key_id, label=label)


def _unauthorized(message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=401)


def _bad_author() -> JSONResponse:
    return JSONResponse(
        {"error": "missing or malformed X-Author header: an author slug is required"},
        status_code=400,
    )


def author_header(value: str | None) -> str | None:
    """The author a request declares, or None when it is missing, malformed, or reserved."""
    if value is None or not AUTHOR_RE.fullmatch(value) or value == RESERVED_AUTHOR:
        return None
    return value


class ApiKeyAuthMiddleware:
    """Fail-closed X-API-Key and X-Author gate; /health* is exempt from both."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["path"].startswith(EXEMPT_PATH_PREFIX):
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)
        api_key = request.headers.get(API_KEY_HEADER)
        identity = await authenticate_request(api_key) if api_key else None
        if identity is None:
            response = _unauthorized("missing, unknown, or revoked X-API-Key")
            await response(scope, receive, send)
            return
        author = author_header(request.headers.get(AUTHOR_HEADER))
        if author is None:
            response = _bad_author()
            await response(scope, receive, send)
            return
        request.state.key = identity
        request.state.author = author
        await self.app(scope, receive, send)
