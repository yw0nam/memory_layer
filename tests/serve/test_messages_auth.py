"""Claim and cancel outcomes: success, 404 for an unknown id, 409 for a non-pending one.

Drives memory_base.serve.messages.store.claim_message / cancel_message against a
fake connection, without a database.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from memory_base.serve.messages import store

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
    """Answers the conditional UPDATE with `updated` and the existence lookup with `exists`."""

    def __init__(self, updated, exists=True):
        self._updated = updated
        self._exists = exists

    async def fetchrow(self, query, *args):
        return self._updated

    async def fetchval(self, query, *args):
        return 1 if self._exists else None


def _patch_acquire(monkeypatch, conn):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(store.db, "acquire", acquire)
    monkeypatch.setattr(store, "ensure_schema_once", _noop)


async def _noop(conn):
    return None


@pytest.mark.parametrize("action", [store.claim_message, store.cancel_message])
def test_a_pending_message_is_actionable(monkeypatch, action):
    message_id = uuid.uuid4()
    _patch_acquire(monkeypatch, FakeConn(_row(id=message_id)))
    assert asyncio.run(action(message_id))["status"] == "info"


@pytest.mark.parametrize("action", [store.claim_message, store.cancel_message])
def test_an_unknown_id_is_404(monkeypatch, action):
    _patch_acquire(monkeypatch, FakeConn(None, exists=False))
    with pytest.raises(store.MessageNotFound):
        asyncio.run(action(uuid.uuid4()))


@pytest.mark.parametrize("action", [store.claim_message, store.cancel_message])
def test_a_message_that_is_no_longer_pending_is_409(monkeypatch, action):
    _patch_acquire(monkeypatch, FakeConn(None))
    with pytest.raises(store.MessageConflict):
        asyncio.run(action(uuid.uuid4()))
