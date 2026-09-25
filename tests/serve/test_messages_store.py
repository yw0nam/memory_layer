"""Storage contracts for the message lane, driven against a recording fake connection.

Postgres enforces at-most-once claims, snapshot supersede, and the namespace
guard through the statements themselves (row locks, an advisory lock, and
conditional UPDATEs), so these tests pin those statements and their order.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest

from memory_base.serve import auth, messages, namespaces

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

    def __init__(self, *, registered=True, lookups=(), insert_error=None, updated=None):
        self.registered = registered
        self.lookups = list(lookups)
        self.insert_error = insert_error
        self.updated = updated
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
        if query.lstrip().startswith("UPDATE"):
            return self.updated
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

        monkeypatch.setattr(messages.db, "acquire", acquire)
        monkeypatch.setattr(messages, "ensure_schema_once", _noop)
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
    return asyncio.run(messages.send_message(KEY, **fields))


def _handoff(**overrides):
    fields = {
        "status": "in_progress",
        "next": "carry on",
        "scope": "repo:github.com/o/r",
    }
    fields.update(overrides)
    fields["next_text"] = fields.pop("next")
    return _send(**fields)


# ---- send ---------------------------------------------------------------------


def test_handoff_send_locks_the_chain_then_inserts_then_supersedes_older_pending(use):
    conn = use(RecordingConn())
    row, replay = _handoff()

    assert replay is False
    kinds = [s[0].split()[0] for s in conn.statements]
    assert kinds == ["SELECT", "SELECT", "INSERT", "UPDATE"]
    guard, lock, insert, supersede = conn.statements
    assert "FOR SHARE" in guard[0] and guard[1] == ("default",)
    assert "pg_advisory_xact_lock" in lock[0]
    assert lock[1] == ("default\x1frepo:github.com/o/r\x1fdeploy plan",)
    assert "clock_timestamp()" in insert[0]
    assert "SET superseded_at = clock_timestamp()" in supersede[0]
    assert "purpose = 'handoff' AND scope = $2 AND subject_key = $3 AND id <> $4" in supersede[0]
    assert "claimed_at IS NULL AND cancelled_at IS NULL AND superseded_at IS NULL" in supersede[0]
    assert supersede[1] == ("default", "repo:github.com/o/r", "deploy plan", uuid.UUID(row["id"]))
    # The guard, the lock, and the supersede share one transaction with the insert.
    assert all(depth >= 1 for _, _, depth in conn.statements)


def test_general_message_takes_no_lock_and_supersedes_nothing(use):
    conn = use(RecordingConn())
    row, _ = _send()

    assert row["purpose"] == "message"
    assert conn.sql("pg_advisory_xact_lock") == []
    assert conn.sql("superseded_at") == []


def test_send_into_unregistered_namespace_writes_nothing(use):
    conn = use(RecordingConn(registered=False))
    with pytest.raises(namespaces.NamespaceError):
        _send(namespace="ghost")
    assert conn.sql("INSERT") == []


def test_send_writes_only_the_messages_table(use):
    conn = use(RecordingConn())
    _handoff()
    for sql, _, _ in conn.statements:
        for searched in ("memory_chunks", "code_chunks", "doc_rows"):
            assert searched not in sql


def test_identical_idempotent_retry_replays_without_inserting(use):
    first = use(RecordingConn(lookups=[None]))
    original, _ = _send(idempotency_key="k1")
    stored = _row(
        id=uuid.UUID(original["id"]),
        subject="Deploy Plan",
        subject_key="deploy plan",
        content=original["content"],
        idempotency_key="k1",
    )
    assert first.sql("INSERT")

    conn = use(RecordingConn(lookups=[stored]))
    row, replay = _send(idempotency_key="k1")
    assert replay is True
    assert row["id"] == original["id"]
    assert conn.sql("INSERT") == []


def test_reusing_an_idempotency_key_for_different_content_conflicts(use):
    stored = _row(subject_key="deploy plan", content="# other", idempotency_key="k1")
    conn = use(RecordingConn(lookups=[stored]))
    with pytest.raises(messages.MessageConflict):
        _send(idempotency_key="k1")
    assert conn.sql("INSERT") == []


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


# ---- claim / cancel -------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "stamp"),
    [(messages.claim_message, "claimed_at"), (messages.cancel_message, "cancelled_at")],
)
def test_claim_and_cancel_are_one_conditional_update_checked_at_wake_up(use, action, stamp):
    message_id = uuid.uuid4()
    owner = _row(id=message_id, sender_key=KEY.key_id)
    conn = use(RecordingConn(lookups=[owner], updated=_row(id=message_id)))

    asyncio.run(action(message_id, KEY))

    (update,) = conn.sql("UPDATE")
    assert f"SET {stamp} = clock_timestamp()" in update[0]
    assert "claimed_at IS NULL AND cancelled_at IS NULL AND superseded_at IS NULL" in update[0]
    assert "expires_at > clock_timestamp()" in update[0]


def test_a_claim_the_conditional_update_refuses_is_a_conflict(use):
    message_id = uuid.uuid4()
    use(RecordingConn(lookups=[_row(id=message_id)], updated=None))
    with pytest.raises(messages.MessageConflict):
        asyncio.run(messages.claim_message(message_id, KEY))


# ---- list -----------------------------------------------------------------------


def test_listing_reads_only_pending_unexpired_rows_newest_first(use):
    conn = use(RecordingConn())
    asyncio.run(
        messages.list_messages(
            namespaces=["default"],
            purpose="handoff",
            scope="repo:https://GitHub.com/o/r.git",
            subject="  Deploy   Plan ",
            limit=5,
        )
    )
    ((sql, args, _),) = conn.statements
    assert "claimed_at IS NULL AND cancelled_at IS NULL AND superseded_at IS NULL" in sql
    assert "expires_at > now()" in sql
    assert "ORDER BY created_at DESC, id DESC" in sql
    assert args == (["default"], "handoff", "repo:github.com/o/r", "deploy plan", 5)


# ---- purge ----------------------------------------------------------------------


def test_member_purge_resolves_ownership_inside_the_delete(use):
    conn = use(RecordingConn())
    assert asyncio.run(messages.delete_terminal_messages("alice")) == 4
    ((sql, args, _),) = conn.statements
    assert sql.startswith("DELETE FROM")
    assert "namespace IN (SELECT name FROM" in sql and "WHERE owner = $1" in sql
    assert args == ("alice",)


def test_admin_purge_covers_every_namespace_but_only_terminal_rows(use):
    conn = use(RecordingConn())
    asyncio.run(messages.delete_terminal_messages(None))
    ((sql, args, _),) = conn.statements
    assert "owner" not in sql
    assert args == ()
    for terminal in (
        "claimed_at IS NOT NULL",
        "cancelled_at IS NOT NULL",
        "superseded_at IS NOT NULL",
        "expires_at <= now()",
    ):
        assert terminal in sql
