"""save_note holds its namespace row inside the transaction that inserts the note."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from memory_base.serve.access import namespaces
from memory_base.serve.notes import store


class _Tx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.depth += 1

    async def __aexit__(self, *args):
        self.conn.depth -= 1


class RecordingConn:
    """Records (sql, args, transaction depth); answers the lookups save_note makes."""

    def __init__(self, *, registered=True):
        self.registered = registered
        self.statements: list[tuple[str, tuple, int]] = []
        self.depth = 0

    def transaction(self):
        return _Tx(self)

    def _record(self, query, args):
        self.statements.append((" ".join(query.split()), args, self.depth))

    async def fetchval(self, query, *args):
        self._record(query, args)
        if "FOR SHARE" in query:
            return 1 if self.registered else None
        return True

    async def fetch(self, query, *args):
        self._record(query, args)
        return []

    async def execute(self, query, *args):
        self._record(query, args)
        if "INSERT INTO" in query:
            return "INSERT 0 1"
        return "UPDATE 1"

    def sql(self, needle):
        return [s for s in self.statements if needle in s[0]]


async def _noop(conn):
    return None


async def _embed(embedder, text):
    return [0.0] * 4


@pytest.fixture
def use(monkeypatch):
    def _use(conn):
        @asynccontextmanager
        async def acquire(timeout=None):
            yield conn

        monkeypatch.setattr(store.db, "acquire", acquire)
        monkeypatch.setattr(store, "ensure_schema_once", _noop)
        monkeypatch.setattr(store, "embed_text", _embed)
        return conn

    return _use


def _save(**overrides):
    fields = {"tags": ["test"], "kind": "work", "namespace": "team-a"}
    fields.update(overrides)
    return asyncio.run(store.save_note("prefer ruff for linting", **fields))


def test_save_note_holds_the_namespace_row_inside_its_transaction(use):
    conn = use(RecordingConn())
    result = _save()

    assert result["stored"] is True
    guard, _similar, insert = conn.statements
    assert "FOR SHARE" in guard[0] and guard[1] == ("team-a",)
    assert guard[2] >= 1
    assert "INSERT INTO" in insert[0]
    assert insert[2] >= 1


def test_save_note_into_an_unregistered_namespace_writes_nothing(use):
    conn = use(RecordingConn(registered=False))
    with pytest.raises(namespaces.NamespaceError):
        _save()
    assert conn.sql("INSERT") == []


def test_a_supersede_records_the_replacement_on_the_replaced_note(use):
    conn = use(RecordingConn())
    result = _save(supersedes="note:team-a:old", author="claude-code")
    [archive] = conn.sql("SET archived_at")
    assert "'replaced_by'" in archive[0] and "'archived_by'" in archive[0]
    assert result["id"] in archive[1]
    assert "note:team-a:old" in archive[1]
    assert archive[2] >= 1
