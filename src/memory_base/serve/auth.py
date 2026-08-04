"""API-key authentication and the per-key namespace visibility model.

Every REST request (except /health*) must carry a valid ``X-API-Key`` header;
ApiKeyAuthMiddleware looks it up (by sha256 hash) and attaches a KeyIdentity
to ``request.state.key`` for route handlers to enforce namespace access.
Fail-closed: no key, an unknown key, or a revoked key all get 401.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once

API_KEY_HEADER = "x-api-key"
EXEMPT_PATH_PREFIX = "/health"


def hash_key(plaintext: str) -> str:
    """The sha256 hex digest stored in api_keys.key_hash; plaintext is never stored."""
    return hashlib.sha256(plaintext.encode()).hexdigest()


def generate_key() -> str:
    """A fresh random plaintext API key."""
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class KeyIdentity:
    """A request's authenticated caller: an admin sees every namespace."""

    label: str
    home: str
    is_admin: bool
    allowed: frozenset[str]

    def permits(self, namespace: str) -> bool:
        return self.is_admin or namespace in self.allowed

    def permits_all(self, namespaces: set[str]) -> bool:
        return self.is_admin or namespaces <= self.allowed


async def _owned_or_public_namespaces(conn, label: str) -> frozenset[str]:
    rows = await conn.fetch(
        f"SELECT name FROM \"{PG_SCHEMA}\".namespaces WHERE visibility = 'public' OR owner = $1",
        label,
    )
    return frozenset(row["name"] for row in rows)


async def authenticate_request(plaintext_key: str) -> KeyIdentity | None:
    """Look up a plaintext key; None for missing/unknown/revoked."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        row = await conn.fetchrow(
            f'SELECT label, home, is_admin FROM "{PG_SCHEMA}".api_keys '
            "WHERE key_hash = $1 AND revoked_at IS NULL",
            hash_key(plaintext_key),
        )
        if row is None:
            return None
        allowed = (
            frozenset()
            if row["is_admin"]
            else await _owned_or_public_namespaces(conn, row["label"])
        )
        return KeyIdentity(
            label=row["label"], home=row["home"], is_admin=row["is_admin"], allowed=allowed
        )


def _unauthorized(message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=401)


class ApiKeyAuthMiddleware:
    """Fail-closed X-API-Key gate; /health* is exempt from authentication."""

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
        request.state.key = identity
        await self.app(scope, receive, send)
