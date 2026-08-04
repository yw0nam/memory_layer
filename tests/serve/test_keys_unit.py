"""Unit tests for the memory_base.serve.keys CLI: no real DB.

memory_base.core.db.acquire is monkeypatched to a fake connection, matching
the convention in tests/serve/test_namespaces_unit.py.
"""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager

import pytest

from memory_base.serve import keys


class FakeConnection:
    def __init__(self, fetchrow_results=None):
        self._fetchrow = list(fetchrow_results or [])
        self.executed: list[tuple] = []

    async def fetchrow(self, query, *args):
        return self._fetchrow.pop(0)

    async def execute(self, query, *args):
        self.executed.append((query, args))
        return "UPDATE 1"


def _patch_acquire(monkeypatch, conn):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(keys.db, "acquire", acquire)

    async def _noop_ensure_schema_once(conn):
        return None

    monkeypatch.setattr(keys, "ensure_schema_once", _noop_ensure_schema_once)


# ---- new_key ------------------------------------------------------------------


def test_new_key_stores_sha256_hash_not_plaintext(monkeypatch):
    conn = FakeConnection(fetchrow_results=[{"visibility": "public", "owner": None}])
    _patch_acquire(monkeypatch, conn)
    plaintext = asyncio.run(keys.new_key("alice"))
    assert len(plaintext) > 20
    insert_query, insert_args = conn.executed[0]
    assert "INSERT INTO" in insert_query
    stored_hash = insert_args[0]
    assert stored_hash == hashlib.sha256(plaintext.encode()).hexdigest()
    assert plaintext not in insert_query
    assert stored_hash != plaintext


def test_new_key_defaults_home_to_default_and_non_admin(monkeypatch):
    conn = FakeConnection(fetchrow_results=[{"visibility": "public", "owner": None}])
    _patch_acquire(monkeypatch, conn)
    asyncio.run(keys.new_key("alice"))
    _, args = conn.executed[0]
    assert args[1] == "alice"
    assert args[2] == "default"
    assert args[3] is False


def test_new_key_rejects_unknown_home(monkeypatch):
    conn = FakeConnection(fetchrow_results=[None])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(keys.HomeNamespaceError):
        asyncio.run(keys.new_key("alice", home="ghost"))
    assert conn.executed == []


def test_new_key_rejects_private_home_not_owned_by_label(monkeypatch):
    conn = FakeConnection(fetchrow_results=[{"visibility": "private", "owner": "bob"}])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(keys.HomeNamespaceError):
        asyncio.run(keys.new_key("alice", home="team-b"))
    assert conn.executed == []


def test_new_key_allows_private_home_owned_by_label(monkeypatch):
    conn = FakeConnection(fetchrow_results=[{"visibility": "private", "owner": "alice"}])
    _patch_acquire(monkeypatch, conn)
    plaintext = asyncio.run(keys.new_key("alice", home="team-a"))
    assert plaintext
    assert conn.executed


def test_new_key_admin_bypasses_private_home_ownership(monkeypatch):
    conn = FakeConnection(fetchrow_results=[{"visibility": "private", "owner": "bob"}])
    _patch_acquire(monkeypatch, conn)
    plaintext = asyncio.run(keys.new_key("alice", home="team-b", is_admin=True))
    assert plaintext
    assert conn.executed


# ---- revoke_key -----------------------------------------------------------------


def test_revoke_key_matches_by_prefix(monkeypatch):
    conn = FakeConnection()
    _patch_acquire(monkeypatch, conn)
    asyncio.run(keys.revoke_key("abcd1234"))
    query, args = conn.executed[0]
    assert "SET revoked_at = now()" in query
    assert "LIKE $1" in query
    assert args[0] == "abcd1234"


@pytest.mark.parametrize("prefix", ["", "abcdefg"])
def test_revoke_key_rejects_prefix_shorter_than_eight_characters(monkeypatch, prefix):
    conn = FakeConnection()
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(ValueError, match="at least 8 characters"):
        asyncio.run(keys.revoke_key(prefix))
    assert conn.executed == []


# ---- CLI parsing ------------------------------------------------------------------


def test_cli_new_requires_label():
    with pytest.raises(SystemExit):
        keys.build_parser().parse_args(["new"])


def test_cli_new_parses_home_and_admin_flag():
    args = keys.build_parser().parse_args(["new", "alice", "--home", "team-a", "--admin"])
    assert args.command == "new"
    assert args.label == "alice"
    assert args.home == "team-a"
    assert args.admin is True


def test_cli_revoke_parses_prefix():
    args = keys.build_parser().parse_args(["revoke", "abcd1234"])
    assert args.command == "revoke"
    assert args.prefix_or_hash == "abcd1234"


def test_cli_revoke_reports_short_prefix_cleanly():
    with pytest.raises(SystemExit, match="at least 8 characters"):
        keys.main(["revoke", "short"])


def test_cli_list_parses():
    args = keys.build_parser().parse_args(["list"])
    assert args.command == "list"
