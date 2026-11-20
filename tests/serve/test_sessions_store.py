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
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        self.calls.append(("BEGIN", ()))

    async def __aexit__(self, *args):
        self.calls.append(("COMMIT", ()))


class RecordingConn:
    """Answers the id lookup with the stored rows of the asked author, `max(seq)` with
    `last_seq`, and any other fetch with `stored`."""

    def __init__(self, stored=(), last_seq=None):
        self.stored = list(stored)
        self.last_seq = last_seq or {}
        self.calls: list[tuple[str, tuple]] = []

    def transaction(self):
        return _Tx(self.calls)

    async def execute(self, query, *args):
        self.calls.append((" ".join(query.split()), args))
        return "DELETE 2" if query.lstrip().startswith("DELETE") else "SELECT 1"

    async def fetchval(self, query, *args):
        self.calls.append((" ".join(query.split()), args))
        return 1

    async def fetch(self, query, *args):
        self.calls.append((" ".join(query.split()), args))
        if "max(seq)" in query:
            return [{"session_id": sid, "last": seq} for sid, seq in self.last_seq.items()]
        if "AND id = ANY" in query:
            return [row for row in self.stored if row["author"] == args[0]]
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
    seqs = [(row[2], row[3]) for row in _inserted(conn)]
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
    assert row[0] == str(fresh["id"]) and row[3] == 3


def test_each_author_stores_its_own_copy_of_an_id_and_a_racing_insert_cannot_fail(use):
    event = _event()
    conn = use(RecordingConn(stored=[{**event, "author": "natsume", "seq": 1}]))
    assert _store([event]) == {"stored": 1, "known": 0, "conflicts": []}
    [lookup] = [args for sql, args in conn.calls if "AND id = ANY" in sql]
    assert lookup[0] == "claude-code"
    [insert] = [sql for sql, _ in conn.calls if sql.startswith("INSERT INTO")]
    assert insert.endswith("ON CONFLICT DO NOTHING")


def test_an_id_repeated_inside_one_batch_is_stored_once(use):
    event = _event()
    conn = use(RecordingConn())
    assert _store([event, event]) == {"stored": 1, "known": 1, "conflicts": []}
    assert len(_inserted(conn)) == 1


def _row(seq, content):
    return {"seq": seq, "role": "user", "tool_name": None, "event_at": AT, "content": content}


def _read(**overrides):
    args = dict(author="claude-code", session_id="s1", from_seq=1, limit=50, role=None)
    return asyncio.run(store.read_session(**{**args, "contains": None, **overrides}))


def test_a_read_page_stops_before_200000_content_chars_and_names_the_next_seq(use):
    use(RecordingConn(stored=[_row(n, "x" * 90_000) for n in (1, 2, 3)]))
    page = _read()
    assert [e["seq"] for e in page["events"]] == [1, 2]
    assert page["next_from_seq"] == 3


def test_a_read_page_names_the_next_seq_only_when_more_events_remain(use):
    conn = use(RecordingConn(stored=[_row(n, "x") for n in (4, 5, 6)]))
    page = _read(from_seq=4, limit=2)
    assert ([e["seq"] for e in page["events"]], page["next_from_seq"]) == ([4, 5], 6)
    conn.stored = conn.stored[:2]
    assert _read(from_seq=4, limit=2)["next_from_seq"] is None


def test_a_read_page_always_holds_one_event(use, monkeypatch):
    monkeypatch.setattr(store, "READ_MAX_CHARS", 10)
    use(RecordingConn(stored=[_row(1, "x" * 50), _row(2, "y")]))
    page = _read()
    assert ([e["seq"] for e in page["events"]], page["next_from_seq"]) == ([1], 2)


def test_delete_takes_the_session_lock_inside_a_transaction(use):
    conn = use(RecordingConn())
    assert asyncio.run(store.delete_session(author="claude-code", session_id="s1")) == 2
    steps = [sql.split()[0] for sql, _ in conn.calls]
    assert steps == ["BEGIN", "SELECT", "DELETE", "COMMIT"]
    assert conn.calls[1][1] == ("claude-code\x1fs1",)
