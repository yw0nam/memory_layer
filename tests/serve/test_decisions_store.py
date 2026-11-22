"""Storage contracts of the decision lane that a faked REST layer cannot show.

A recording fake connection checks the lock-then-number order and that a refused
supersede stores nothing; the SQL behaviour itself is in test_decisions_integration.py.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from memory_base.serve.decisions import store

SCOPE = "repo:github.com/o/r"


class _Tx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.depth += 1

    async def __aexit__(self, *args):
        self.conn.depth -= 1


class RecordingConn:
    def __init__(self, *, next_number=1, replaced=None):
        self.next_number = next_number
        self.replaced = replaced
        self.statements: list[tuple[str, tuple, int]] = []
        self.depth = 0

    def transaction(self):
        return _Tx(self)

    async def execute(self, query, *args):
        self.statements.append((" ".join(query.split()), args, self.depth))

    async def fetchval(self, query, *args):
        self.statements.append((" ".join(query.split()), args, self.depth))
        return self.replaced if "UPDATE" in query else self.next_number


@pytest.fixture
def use(monkeypatch):
    async def _noop(conn):
        return None

    def _use(conn):
        @asynccontextmanager
        async def acquire(timeout=None):
            yield conn

        monkeypatch.setattr(store.db, "acquire", acquire)
        monkeypatch.setattr(store, "ensure_schema_once", _noop)
        return conn

    return _use


def _record(**overrides):
    fields = {"author": "claude-code", "scope": SCOPE, "title": "t", "decision": "d"}
    fields.update(overrides)
    return asyncio.run(store.record_decision(**fields))


def test_the_number_is_taken_under_a_lock_keyed_by_scope_and_author(use):
    conn = use(RecordingConn(next_number=4))
    assert _record(scope="repo:https://github.com/o/r.git") == {
        "number": 4,
        "superseded": None,
    }
    statements = [sql for sql, _, _ in conn.statements]
    lock = next(i for i, sql in enumerate(statements) if "pg_advisory_xact_lock" in sql)
    number = next(i for i, sql in enumerate(statements) if "max(number)" in sql)
    assert lock < number
    assert conn.statements[lock][1] == (f"{SCOPE}\x1fclaude-code",)
    assert all(depth == 1 for _, _, depth in conn.statements)


def test_a_refused_supersede_stores_nothing(use):
    conn = use(RecordingConn(replaced=None))
    with pytest.raises(ValueError, match="D-3"):
        _record(supersedes=3)
    assert not [sql for sql, _, _ in conn.statements if "INSERT" in sql]


def test_a_supersede_replaces_the_active_decision_in_the_same_transaction(use):
    conn = use(RecordingConn(next_number=5, replaced=3))
    assert _record(supersedes=3) == {"number": 5, "superseded": 3}
    update = next(s for s in conn.statements if "UPDATE" in s[0])
    assert "status = 'active'" in update[0]
    assert update[1] == (SCOPE, "claude-code", 3, 5)
    assert update[2] == 1
