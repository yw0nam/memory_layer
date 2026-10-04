"""Storage contracts for the message lane that the integration suite cannot reach.

A recording fake connection drives the lost idempotency-insert race and pins the
terminal-row predicate of the admin purge.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest

from memory_base.serve.access import auth
from memory_base.serve.messages import store

NOW = datetime.now(timezone.utc)

KEY = auth.KeyIdentity(
    key_id="sender-key-hash",
    label="sender",
    home="default",
    is_admin=False,
    allowed=frozenset({"default"}),
    authors=frozenset({"claude-code"}),
)


class _Tx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.depth += 1

    async def __aexit__(self, *args):
        self.conn.depth -= 1


class RecordingConn:
    """Records (sql, args, transaction depth); answers from scripted lookups."""

    def __init__(self, *, registered=True, lookups=(), insert_error=None):
        self.registered = registered
        self.lookups = list(lookups)
        self.insert_error = insert_error
        self.statements: list[tuple[str, tuple, int]] = []
        self.depth = 0

    def transaction(self):
        return _Tx(self)

    def _record(self, query, args):
        self.statements.append((" ".join(query.split()), args, self.depth))

    async def fetchval(self, query, *args):
        self._record(query, args)
        return 1 if self.registered else None

    async def fetchrow(self, query, *args):
        self._record(query, args)
        if "INSERT INTO" in query:
            if self.insert_error is not None:
                raise self.insert_error
            return _row(
                id=args[0],
                namespace=args[1],
                purpose=args[2],
                scope=args[3],
                subject=args[4],
                subject_key=args[5],
                status=args[6],
                content=args[7],
                author=args[8],
                expires_at=args[11],
            )
        return self.lookups.pop(0)

    async def fetch(self, query, *args):
        self._record(query, args)
        return []

    async def execute(self, query, *args):
        self._record(query, args)
        return "DELETE 4"

    def sql(self, needle):
        return [s for s in self.statements if needle in s[0]]


def _row(**overrides):
    row = {
        "id": uuid.uuid4(),
        "namespace": "default",
        "purpose": "message",
        "scope": None,
        "subject": "s",
        "subject_key": "s",
        "status": "info",
        "content": "# s",
        "author": "claude-code",
        "idempotency_key": None,
        "created_at": NOW,
        "expires_at": NOW + timedelta(days=1),
    }
    row.update(overrides)
    return row


async def _noop(conn):
    return None


@pytest.fixture
def use(monkeypatch):
    def _use(conn):
        @asynccontextmanager
        async def acquire(timeout=None):
            yield conn

        monkeypatch.setattr(store.db, "acquire", acquire)
        monkeypatch.setattr(store, "ensure_schema_once", _noop)
        return conn

    return _use


def _send(**overrides):
    fields = {
        "namespace": "default",
        "author": "claude-code",
        "subject": "Deploy Plan",
        "status": "info",
        "result": "r",
    }
    fields.update(overrides)
    return asyncio.run(store.send_message(KEY, **fields))


def test_losing_the_idempotency_insert_race_replays_the_winner(use):
    probe = use(RecordingConn(lookups=[None]))
    winner_row, _ = _send(idempotency_key="k1")
    assert probe.sql("INSERT")
    winner = _row(
        id=uuid.UUID(winner_row["id"]),
        subject="Deploy Plan",
        subject_key="deploy plan",
        content=winner_row["content"],
        idempotency_key="k1",
    )
    conn = use(
        RecordingConn(
            lookups=[None, winner], insert_error=asyncpg.UniqueViolationError("duplicate key")
        )
    )
    row, replay = _send(idempotency_key="k1")
    assert replay is True
    assert row["id"] == winner_row["id"]
    assert len(conn.sql("idempotency_key = $2")) == 2


def test_admin_purge_covers_every_namespace_but_only_terminal_rows(use):
    conn = use(RecordingConn())
    asyncio.run(store.delete_terminal_messages(None))
    ((sql, args, _),) = conn.statements
    assert "owner" not in sql
    assert args == ()
    for terminal in (
        "claimed_at IS NOT NULL",
        "cancelled_at IS NOT NULL",
        "superseded_at IS NOT NULL",
        "(expires_at IS NOT NULL AND expires_at <= now())",
    ):
        assert terminal in sql
