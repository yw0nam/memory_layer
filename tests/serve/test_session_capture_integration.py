"""Session capture end to end against the session's throwaway Postgres.

Uploads two sessions, lists them newest first within a time window, reads one
back filtered by a substring, and re-uploads it with two more turns.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api, auth, namespaces

pytestmark = pytest.mark.integration

NAMESPACE = "capture-it"
TURNS = [
    {"role": "user", "text": "Our staging cluster is three ARM nodes behind one HAProxy."},
    {"role": "assistant", "text": "Then every staging image must also be built for arm64."},
    {"role": "user", "text": "And deploys drop connections unless the backend is drained."},
    {"role": "assistant", "text": "So the deploy script drains the HAProxy backend first."},
]
MORE = [
    {"role": "user", "text": "The nightly backup now runs at 04:00 on the storage host."},
    {"role": "assistant", "text": "Noted: backups at 04:00, on the storage host."},
]
OLDER_TURNS = [
    {"role": "user", "text": "Which host keeps the offsite backup copies?"},
    {"role": "assistant", "text": "The storage host in the second rack."},
]

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


async def _cleanup() -> None:
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".conversation_sources WHERE namespace = $1', NAMESPACE
        )
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".namespaces WHERE name = $1', NAMESPACE)
    finally:
        await conn.close()


@pytest.fixture
def namespace(monkeypatch):
    identity = auth.KeyIdentity(
        key_id="it-capture-key",
        label="capture-it",
        home=NAMESPACE,
        is_admin=True,
        allowed=frozenset(),
        authors=frozenset({"claude-code"}),
    )

    async def authenticate(plaintext_key):
        return identity if plaintext_key == "test-key" else None

    monkeypatch.setattr(auth, "authenticate_request", authenticate)
    asyncio.run(_cleanup())
    asyncio.run(namespaces.create_namespace(NAMESPACE))
    yield NAMESPACE
    asyncio.run(_cleanup())


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _listed(**params) -> list[dict]:
    response = client.get("/conversations", params={"namespace": NAMESPACE, **params})
    assert response.status_code == 200, response.json()
    return response.json()


def test_captured_sessions_are_listed_by_time_and_searched_within_one(namespace):
    now = time.time()
    newer = {
        "namespace": namespace,
        "origin": "claude_code",
        "external_session_id": "capture-it-newer",
        "started_at": now - 600,
        "ended_at": now - 60,
        "turns": TURNS,
        "metadata": {"repo": "infra", "cwd": "/srv/infra"},
    }
    older = {
        **newer,
        "external_session_id": "capture-it-older",
        "started_at": now - 7200,
        "ended_at": now - 7000,
        "turns": OLDER_TURNS,
        "metadata": {},
    }
    stored = client.post("/conversations", json=newer)
    assert stored.status_code == 201, stored.json()
    newer_id = stored.json()["id"]
    assert stored.json() == {"id": newer_id, "created": True, "turns": 4}
    stored_older = client.post("/conversations", json=older)
    assert stored_older.status_code == 201, stored_older.json()
    older_id = stored_older.json()["id"]

    listed = _listed(origin="claude_code")
    assert [row["id"] for row in listed] == [newer_id, older_id]
    assert listed[0] == {
        "id": newer_id,
        "namespace": namespace,
        "origin": "claude_code",
        "external_session_id": "capture-it-newer",
        "started_at": newer["started_at"],
        "ended_at": newer["ended_at"],
        "turn_count": 4,
        "repo": "infra",
        "preview": TURNS[0]["text"],
    }
    assert listed[1]["repo"] is None
    assert listed[1]["preview"] == OLDER_TURNS[0]["text"]
    assert _listed(origin="hermes") == []
    assert [row["id"] for row in _listed(since=_iso(now - 3600))] == [newer_id]
    assert [row["id"] for row in _listed(until=_iso(now - 3600))] == [older_id]
    assert [row["id"] for row in _listed(limit=1)] == [newer_id]

    matched = client.get(f"/conversations/{newer_id}", params={"contains": "haproxy"})
    assert matched.status_code == 200, matched.json()
    assert matched.json()["turns"] == [{"index": 0, **TURNS[0]}, {"index": 3, **TURNS[3]}]
    missed = client.get(f"/conversations/{newer_id}", params={"contains": "kubernetes"})
    assert missed.status_code == 200
    assert missed.json()["turns"] == []

    extended = client.post(
        "/conversations", json={**newer, "turns": [*TURNS, *MORE], "ended_at": now}
    )
    assert extended.status_code == 200, extended.json()
    assert extended.json() == {"id": newer_id, "created": False, "turns": 6}
    assert _listed(origin="claude_code")[0]["turn_count"] == 6
