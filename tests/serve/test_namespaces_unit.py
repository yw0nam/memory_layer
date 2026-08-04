"""Unit tests for the namespace registry (memory_base.serve.namespaces): no real DB.

memory_base.core.db.acquire is monkeypatched to a fake connection, matching the
convention already used in tests/serve/test_ingest_api.py
(``monkeypatch.setattr(ingest_api.db, "acquire", acquire)``).

Collection fails today: memory_base.serve.namespaces does not exist yet.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from memory_base.serve import namespaces


class FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(self, fetchval_results=None, fetch_results=None):
        self._fetchval = list(fetchval_results or [])
        self._fetch = list(fetch_results or [])
        self.queries: list = []

    def transaction(self):
        return FakeTransaction()

    async def fetchval(self, query, *args):
        self.queries.append((query, args))
        return self._fetchval.pop(0)

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self._fetch.pop(0)

    async def execute(self, query, *args):
        self.queries.append((query, args))


def _patch_acquire(monkeypatch, conn):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(namespaces.db, "acquire", acquire)
    monkeypatch.setattr(namespaces, "ensure_schema_once", _noop_ensure_schema_once)


async def _noop_ensure_schema_once(conn):
    return None


# ---- validate_namespace_name (pure) ----------------------------------------


@pytest.mark.parametrize("name", ["default", "team-a", "team_a", "a", "a" * 64])
def test_valid_slugs_accepted(name):
    assert namespaces.validate_namespace_name(name) == name


@pytest.mark.parametrize(
    "name",
    ["", "Team-A", "team a", "team.a", "a" * 65, None, 123, ["team-a"]],
)
def test_invalid_slugs_rejected(name):
    with pytest.raises(namespaces.NamespaceError):
        namespaces.validate_namespace_name(name)


# ---- create_namespace -------------------------------------------------------


def test_create_namespace_rejects_bad_slug_before_touching_db(monkeypatch):
    conn = FakeConnection()
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(namespaces.NamespaceError):
        asyncio.run(namespaces.create_namespace("Bad Name"))
    assert conn.queries == []


def test_create_namespace_inserts_and_returns_name_and_created_at(monkeypatch):
    conn = FakeConnection(fetchval_results=["team-a"])
    _patch_acquire(monkeypatch, conn)
    result = asyncio.run(namespaces.create_namespace("team-a"))
    assert result["name"] == "team-a"
    assert isinstance(result["created_at"], float)
    sql, args = conn.queries[0]
    assert "INSERT INTO" in sql and "ON CONFLICT (name) DO NOTHING" in sql
    assert args[0] == "team-a"


def test_create_namespace_duplicate_raises_exists_error(monkeypatch):
    conn = FakeConnection(fetchval_results=[None])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(namespaces.NamespaceExistsError):
        asyncio.run(namespaces.create_namespace("team-a"))


def test_create_namespace_defaults_to_public_no_owner(monkeypatch):
    conn = FakeConnection(fetchval_results=["team-a"])
    _patch_acquire(monkeypatch, conn)
    result = asyncio.run(namespaces.create_namespace("team-a"))
    assert result["visibility"] == "public"
    assert result["owner"] is None
    sql, args = conn.queries[0]
    assert args[2] == "public"
    assert args[3] is None


def test_create_namespace_private_records_owner(monkeypatch):
    conn = FakeConnection(fetchval_results=["team-a"])
    _patch_acquire(monkeypatch, conn)
    result = asyncio.run(namespaces.create_namespace("team-a", "private", "alice"))
    assert result["visibility"] == "private"
    assert result["owner"] == "alice"
    sql, args = conn.queries[0]
    assert args[2] == "private"
    assert args[3] == "alice"


def test_create_namespace_bad_visibility_rejected_before_touching_db(monkeypatch):
    conn = FakeConnection()
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(namespaces.NamespaceError, match="visibility"):
        asyncio.run(namespaces.create_namespace("team-a", "hidden"))
    assert conn.queries == []


# ---- get_namespace -----------------------------------------------------------


def test_get_namespace_returns_row(monkeypatch):
    row = {"name": "team-a", "created_at": 5.0, "visibility": "private", "owner": "alice"}
    conn = FakeConnection(fetch_results=[])
    conn.fetchrow_results = [row]

    async def fetchrow(query, *args):
        return conn.fetchrow_results.pop(0)

    conn.fetchrow = fetchrow
    _patch_acquire(monkeypatch, conn)
    assert asyncio.run(namespaces.get_namespace("team-a")) == row


def test_get_namespace_returns_none_when_missing(monkeypatch):
    conn = FakeConnection()

    async def fetchrow(query, *args):
        return None

    conn.fetchrow = fetchrow
    _patch_acquire(monkeypatch, conn)
    assert asyncio.run(namespaces.get_namespace("ghost")) is None


# ---- list_namespaces ---------------------------------------------------------


def test_list_namespaces_returns_rows(monkeypatch):
    rows = [{"name": "default", "created_at": 0.0}, {"name": "team-a", "created_at": 5.0}]
    conn = FakeConnection(fetch_results=[rows])
    _patch_acquire(monkeypatch, conn)
    assert asyncio.run(namespaces.list_namespaces()) == rows


# ---- require_registered / namespace_exists -----------------------------------


def test_require_registered_passes_when_present(monkeypatch):
    conn = FakeConnection(fetchval_results=[True])
    asyncio.run(namespaces.require_registered(conn, "default"))


def test_require_registered_raises_when_absent(monkeypatch):
    conn = FakeConnection(fetchval_results=[False])
    with pytest.raises(namespaces.NamespaceError, match="unregistered namespace"):
        asyncio.run(namespaces.require_registered(conn, "ghost"))


def test_namespace_exists_true(monkeypatch):
    conn = FakeConnection(fetchval_results=[True])
    _patch_acquire(monkeypatch, conn)
    assert asyncio.run(namespaces.namespace_exists("default")) is True


def test_namespace_exists_false(monkeypatch):
    conn = FakeConnection(fetchval_results=[False])
    _patch_acquire(monkeypatch, conn)
    assert asyncio.run(namespaces.namespace_exists("ghost")) is False


# ---- delete_namespace ---------------------------------------------------------


def test_delete_default_is_reserved_without_touching_db(monkeypatch):
    conn = FakeConnection()
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(namespaces.NamespaceReservedError):
        asyncio.run(namespaces.delete_namespace("default"))
    assert conn.queries == []


def test_delete_unknown_namespace_404_equivalent(monkeypatch):
    conn = FakeConnection(fetchval_results=[False])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(namespaces.NamespaceNotFoundError):
        asyncio.run(namespaces.delete_namespace("ghost"))


def test_delete_non_empty_namespace_conflict(monkeypatch):
    conn = FakeConnection(fetchval_results=[True, True])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(namespaces.NamespaceNotEmptyError):
        asyncio.run(namespaces.delete_namespace("team-a"))


def test_delete_empty_namespace_succeeds(monkeypatch):
    conn = FakeConnection(fetchval_results=[True, False])
    _patch_acquire(monkeypatch, conn)
    asyncio.run(namespaces.delete_namespace("team-a"))
    delete_query = next(q for q, _ in conn.queries if "DELETE FROM" in q)
    assert "namespaces" in delete_query
