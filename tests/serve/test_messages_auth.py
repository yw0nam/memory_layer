"""Authorization contracts for claiming and cancelling messages: who may act.

Drives memory_base.serve.messages.claim_message / cancel_message against a
fake connection, so the permission decisions (namespace access for claims,
sender-or-admin for cancels, 404 for anything invisible) are pinned without a
database.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from memory_base.serve import auth, messages

NOW = datetime.now(timezone.utc)


def _row(**overrides):
    row = {
        "id": uuid.uuid4(),
        "namespace": "default",
        "purpose": "message",
        "scope": None,
        "subject": "s",
        "subject_key": "s",
        "status": "info",
        "content": "# S",
        "author": "claude-code",
        "sender_key": "sender-key-hash",
        "idempotency_key": None,
        "created_at": NOW,
        "claimed_at": None,
        "cancelled_at": None,
        "superseded_at": None,
        "expires_at": NOW + timedelta(days=1),
    }
    row.update(overrides)
    return row


class FakeConn:
    """Hands out scripted fetchrow answers, tagged by the SQL kind."""

    def __init__(self, selects=(), updates=()):
        self._selects = list(selects)
        self._updates = list(updates)
        self.executed: list[tuple] = []

    async def fetchrow(self, query, *args):
        if "SELECT" in query:
            return self._selects.pop(0)
        return self._updates.pop(0)

    async def execute(self, query, *args):
        self.executed.append((query, args))

    def transaction(self):
        return _FakeTransaction()


class _FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


def _patch_acquire(monkeypatch, conn):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(messages.db, "acquire", acquire)


def _identity(label="eve", key_id="eve-key-hash", is_admin=False, allowed=("default",)):
    return auth.KeyIdentity(
        key_id=key_id,
        label=label,
        home="default",
        is_admin=is_admin,
        allowed=frozenset(allowed),
        authors=frozenset({"claude-code"}),
    )


# ---- claim authorization ------------------------------------------------------


def test_claim_by_namespace_authorized_member_succeeds(monkeypatch):
    message_id = uuid.uuid4()
    conn = FakeConn(
        selects=[_row(id=message_id)],
        updates=[_row(id=message_id, claimed_at=NOW)],
    )
    _patch_acquire(monkeypatch, conn)
    row = asyncio.run(messages.claim_message(message_id, _identity()))
    assert row["status"] == "info"


def test_claim_out_of_scope_namespace_404(monkeypatch):
    message_id = uuid.uuid4()
    conn = FakeConn(selects=[_row(id=message_id, namespace="team-b")])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(messages.MessageNotFound):
        asyncio.run(messages.claim_message(message_id, _identity(allowed=("default",))))


def test_claim_unknown_id_404(monkeypatch):
    conn = FakeConn(selects=[None])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(messages.MessageNotFound):
        asyncio.run(messages.claim_message(uuid.uuid4(), _identity()))


def test_claim_known_but_not_pending_409(monkeypatch):
    message_id = uuid.uuid4()
    conn = FakeConn(selects=[_row(id=message_id)], updates=[None])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(messages.MessageConflict):
        asyncio.run(messages.claim_message(message_id, _identity()))


def test_claim_admin_may_claim_any_accessible_namespace(monkeypatch):
    message_id = uuid.uuid4()
    conn = FakeConn(
        selects=[_row(id=message_id, namespace="team-b")],
        updates=[_row(id=message_id, claimed_at=NOW)],
    )
    _patch_acquire(monkeypatch, conn)
    row = asyncio.run(messages.claim_message(message_id, _identity(is_admin=True)))
    assert row["status"] == "info"


# ---- cancel authorization -------------------------------------------------------


def test_sender_may_cancel_own_pending(monkeypatch):
    message_id = uuid.uuid4()
    conn = FakeConn(
        selects=[_row(id=message_id)],
        updates=[_row(id=message_id, cancelled_at=NOW)],
    )
    _patch_acquire(monkeypatch, conn)
    row = asyncio.run(messages.cancel_message(message_id, _identity(key_id="sender-key-hash")))
    assert row["status"] == "info"


def test_cancel_by_other_member_404(monkeypatch):
    message_id = uuid.uuid4()
    conn = FakeConn(selects=[_row(id=message_id, sender_key="sender-key-hash")])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(messages.MessageNotFound):
        asyncio.run(messages.cancel_message(message_id, _identity(key_id="eve-key-hash")))


def test_admin_may_cancel_any_pending(monkeypatch):
    message_id = uuid.uuid4()
    conn = FakeConn(
        selects=[_row(id=message_id, sender_key="sender-key-hash")],
        updates=[_row(id=message_id, cancelled_at=NOW)],
    )
    _patch_acquire(monkeypatch, conn)
    row = asyncio.run(messages.cancel_message(message_id, _identity(is_admin=True)))
    assert row["status"] == "info"


def test_sender_without_namespace_visibility_gets_404(monkeypatch):
    """Visibility gates cancel before the sender check: no access, no cancel."""
    message_id = uuid.uuid4()
    conn = FakeConn(selects=[_row(id=message_id, namespace="team-b")])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(messages.MessageNotFound):
        asyncio.run(
            messages.cancel_message(
                message_id, _identity(key_id="sender-key-hash", allowed=("default",))
            )
        )


def test_cancel_non_pending_409(monkeypatch):
    message_id = uuid.uuid4()
    conn = FakeConn(selects=[_row(id=message_id)], updates=[None])
    _patch_acquire(monkeypatch, conn)
    with pytest.raises(messages.MessageConflict):
        asyncio.run(messages.cancel_message(message_id, _identity(key_id="sender-key-hash")))
