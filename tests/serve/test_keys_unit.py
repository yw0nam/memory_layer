"""Unit tests for the memory_base.serve.access.keys CLI: no real DB.

memory_base.core.db.acquire is monkeypatched to a fake connection, matching
the convention in tests/serve/test_namespaces_unit.py.
"""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager

import pytest

from memory_base.serve.access import keys


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


def test_new_key_stores_sha256_hash_not_plaintext_and_the_label(monkeypatch):
    conn = FakeConnection()
    _patch_acquire(monkeypatch, conn)
    plaintext = asyncio.run(keys.new_key("alice"))
    assert len(plaintext) > 20
    insert_query, insert_args = conn.executed[0]
    assert "INSERT INTO" in insert_query
    assert insert_args == (hashlib.sha256(plaintext.encode()).hexdigest(), "alice")
    assert plaintext not in insert_query


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


@pytest.mark.parametrize("flag", ["--home", "--admin", "--author"])
def test_cli_new_takes_nothing_but_a_label(flag):
    with pytest.raises(SystemExit):
        keys.build_parser().parse_args(["new", "alice", flag, "x"])


def test_cli_has_no_authors_command():
    with pytest.raises(SystemExit):
        keys.build_parser().parse_args(["authors", "alice"])


def test_cli_revoke_parses_prefix():
    args = keys.build_parser().parse_args(["revoke", "abcd1234"])
    assert args.command == "revoke"
    assert args.prefix_or_hash == "abcd1234"


def test_cli_revoke_reports_short_prefix_cleanly():
    with pytest.raises(SystemExit, match="at least 8 characters"):
        keys.main(["revoke", "short"])
