"""Unit tests for memory_base.serve.auth: hashing, allowed-set computation,
and ApiKeyAuthMiddleware's 401/health-exempt/identity-attach behavior.

No real DB: memory_base.core.db.acquire is monkeypatched to a fake connection,
matching the convention in tests/serve/test_namespaces_unit.py.
"""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager

import httpx
import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route

from memory_base.serve import auth

pytestmark = pytest.mark.real_auth  # exercises the real auth path; opts out of the tests/serve stub


# ---- hash_key / generate_key -------------------------------------------------


def test_hash_key_is_sha256_hex():
    assert auth.hash_key("secret") == hashlib.sha256(b"secret").hexdigest()


def test_generate_key_is_random_and_url_safe():
    a, b = auth.generate_key(), auth.generate_key()
    assert a != b
    assert len(a) > 20


# ---- KeyIdentity.permits / permits_all ---------------------------------------


def test_admin_permits_any_namespace():
    identity = auth.KeyIdentity(
        key_id="hash", label="alice", home="default", is_admin=True, allowed=frozenset()
    )
    assert identity.permits("anything")
    assert identity.permits_all({"a", "b", "c"})


def test_non_admin_permits_only_allowed_set():
    identity = auth.KeyIdentity(
        key_id="hash",
        label="alice",
        home="default",
        is_admin=False,
        allowed=frozenset({"default", "team-a"}),
    )
    assert identity.permits("team-a")
    assert not identity.permits("team-b")
    assert identity.permits_all({"default", "team-a"})
    assert not identity.permits_all({"default", "team-b"})


# ---- authenticate_request ----------------------------------------------------


class FakeConnection:
    def __init__(self, fetchrow_results=None, fetch_results=None):
        self._fetchrow = list(fetchrow_results or [])
        self._fetch = list(fetch_results or [])

    async def fetchrow(self, query, *args):
        return self._fetchrow.pop(0)

    async def fetch(self, query, *args):
        return self._fetch.pop(0)


def _patch_acquire(monkeypatch, conn):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(auth.db, "acquire", acquire)

    async def _noop_ensure_schema_once(conn):
        return None

    monkeypatch.setattr(auth, "ensure_schema_once", _noop_ensure_schema_once)


def test_authenticate_request_unknown_key_returns_none(monkeypatch):
    conn = FakeConnection(fetchrow_results=[None])
    _patch_acquire(monkeypatch, conn)
    assert asyncio.run(auth.authenticate_request("nope")) is None


def test_authenticate_request_admin_skips_namespace_query(monkeypatch):
    conn = FakeConnection(
        fetchrow_results=[{"label": "alice", "home": "default", "is_admin": True}]
    )
    _patch_acquire(monkeypatch, conn)
    identity = asyncio.run(auth.authenticate_request("plaintext"))
    assert identity.label == "alice"
    assert identity.key_id == auth.hash_key("plaintext")
    assert identity.is_admin is True
    assert conn._fetch == []  # never queried namespaces for an admin


def test_authenticate_request_non_admin_computes_allowed_set(monkeypatch):
    conn = FakeConnection(
        fetchrow_results=[{"label": "alice", "home": "default", "is_admin": False}],
        fetch_results=[[{"name": "default"}, {"name": "team-a"}]],
    )
    _patch_acquire(monkeypatch, conn)
    identity = asyncio.run(auth.authenticate_request("plaintext"))
    assert identity.is_admin is False
    assert identity.allowed == frozenset({"default", "team-a"})


# ---- ApiKeyAuthMiddleware -----------------------------------------------------


async def _whoami(request):
    key = request.state.key
    return JSONResponse({"label": key.label, "is_admin": key.is_admin})


def _app():
    return Starlette(
        routes=[
            Route("/health", lambda r: JSONResponse({"status": "ok"})),
            Route("/whoami", _whoami),
        ],
        middleware=[Middleware(auth.ApiKeyAuthMiddleware)],
    )


def _client(monkeypatch, identity):
    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "good-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()), base_url="http://testserver"
    )


def test_missing_api_key_401(monkeypatch):
    identity = auth.KeyIdentity(
        key_id="hash", label="alice", home="default", is_admin=True, allowed=frozenset()
    )

    async def run():
        async with _client(monkeypatch, identity) as client:
            return await client.get("/whoami")

    response = asyncio.run(run())
    assert response.status_code == 401
    assert "error" in response.json()


def test_unknown_api_key_401(monkeypatch):
    identity = auth.KeyIdentity(
        key_id="hash", label="alice", home="default", is_admin=True, allowed=frozenset()
    )

    async def run():
        async with _client(monkeypatch, identity) as client:
            return await client.get("/whoami", headers={"X-API-Key": "bad-key"})

    response = asyncio.run(run())
    assert response.status_code == 401


def test_revoked_key_returns_none_from_authenticate_and_401(monkeypatch):
    async def fake_authenticate_request(plaintext_key):
        return None  # revoked keys resolve the same as unknown ones

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app()), base_url="http://testserver"
        ) as client:
            return await client.get("/whoami", headers={"X-API-Key": "revoked-key"})

    response = asyncio.run(run())
    assert response.status_code == 401


def test_health_is_exempt_from_authentication(monkeypatch):
    async def unreachable(plaintext_key):
        raise AssertionError("must not authenticate /health")

    monkeypatch.setattr(auth, "authenticate_request", unreachable)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_app()), base_url="http://testserver"
        ) as client:
            return await client.get("/health")

    response = asyncio.run(run())
    assert response.status_code == 200


def test_valid_key_attaches_identity_to_request_state(monkeypatch):
    identity = auth.KeyIdentity(
        key_id="hash",
        label="alice",
        home="default",
        is_admin=False,
        allowed=frozenset({"default"}),
    )

    async def run():
        async with _client(monkeypatch, identity) as client:
            return await client.get("/whoami", headers={"X-API-Key": "good-key"})

    response = asyncio.run(run())
    assert response.status_code == 200
    assert response.json() == {"label": "alice", "is_admin": False}
