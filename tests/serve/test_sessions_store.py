"""Storage contracts of the session record that a faked REST layer cannot show.

A recording fake connection checks the lock-then-number order, the seq assignment and
the known/conflict split; the SQL behaviour itself is in test_sessions_integration.py.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from memory_base.serve.sessions import store

AT = datetime(2026, 5, 1, tzinfo=timezone.utc)


class _Tx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class RecordingConn:
    def __init__(self, stored=(), last_seq=None):
        self.stored = list(stored)
        self.last_seq = last_seq or {}
        self.calls: list[tuple[str, tuple]] = []

    def transaction(self):
        return _Tx()

    async def execute(self, query, *args):
        self.calls.append((" ".join(query.split()), args))

    async def fetch(self, query, *args):
        self.calls.append((" ".join(query.split()), args))
        if "max(seq)" in query:
            return [{"session_id": sid, "last": seq} for sid, seq in self.last_seq.items()]
        return self.stored

    async def executemany(self, query, rows):
        self.calls.append((" ".join(query.split()), tuple(rows)))


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


def _event(session="s1", content="hi", **overrides):
    event = {
        "id": uuid.uuid4(),
        "session_id": session,
        "role": "user",
        "content": content,
        "tool_name": None,
        "tool_use_id": None,
        "event_at": AT,
        "repo": None,
        "host": "mac",
        "cwd": "/w",
    }
    event.update(overrides)
    return event


def _store(events, author="claude-code"):
    return asyncio.run(store.store_events(author, events))


def _inserted(conn):
    [rows] = [args for sql, args in conn.calls if sql.startswith("INSERT INTO")]
    return rows


def test_new_events_take_the_next_seq_per_session_in_request_order_after_the_locks(use):
    conn = use(RecordingConn(last_seq={"s1": 4}))
    events = [_event("s1"), _event("s2"), _event("s1")]
    assert _store(events) == {"stored": 3, "known": 0, "conflicts": []}
    seqs = [(row[3], row[4]) for row in _inserted(conn)]
    assert seqs == [("s1", 5), ("s2", 1), ("s1", 6)]
    statements = [sql for sql, _ in conn.calls]
    locks = [i for i, sql in enumerate(statements) if "pg_advisory_xact_lock" in sql]
    assert len(locks) == 2 and max(locks) < next(
        i for i, sql in enumerate(statements) if "max(seq)" in sql
    )
    assert {args[0] for sql, args in conn.calls if "pg_advisory" in sql} == {
        "claude-code\x1fs1",
        "claude-code\x1fs2",
    }


def test_a_stored_id_is_known_when_the_payload_matches_and_a_conflict_when_it_differs(use):
    same, changed, fresh = _event(content="a"), _event(content="b"), _event(content="c")
    stored = [
        {**same, "author": "claude-code", "seq": 1},
        {**changed, "content": "other", "author": "claude-code", "seq": 2},
    ]
    conn = use(RecordingConn(stored=stored, last_seq={"s1": 2}))
    result = _store([same, changed, fresh])
    assert result == {"stored": 1, "known": 1, "conflicts": [str(changed["id"])]}
    [row] = _inserted(conn)
    assert row[0] == str(fresh["id"]) and row[4] == 3


def test_an_id_stored_by_another_author_is_a_conflict(use):
    event = _event()
    use(RecordingConn(stored=[{**event, "author": "natsume", "seq": 1}]))
    assert _store([event])["conflicts"] == [str(event["id"])]


def test_an_id_repeated_inside_one_batch_is_stored_once(use):
    event = _event()
    conn = use(RecordingConn())
    assert _store([event, event]) == {"stored": 1, "known": 1, "conflicts": []}
    assert len(_inserted(conn)) == 1
