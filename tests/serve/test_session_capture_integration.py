"""Session capture end to end against the session's throwaway Postgres.

Uploads a conversation, runs the distill job it admitted with a fake provider,
and reads the notes back through search; a re-upload with two more turns runs a
second job that sees only the new turns. The embedder is the configured live
endpoint and the content gate is pinned open (tests/conftest.py).
"""

from __future__ import annotations

import asyncio
import time

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api, auth, distill, job_store, namespaces

pytestmark = pytest.mark.integration

NAMESPACE = "capture-it"
KEY_PREFIX = "it-capture-"
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
NOTE_ARCH = "Staging runs on ARM nodes, so every staging container image is built multi-arch."
NOTE_DRAIN = "A staging deploy drains the HAProxy backend first; skipping it drops connections."

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


async def _cleanup() -> None:
    conn = await asyncpg.connect(db_url())
    try:
        for table in ("memory_chunks", "conversation_sources", "jobs"):
            await conn.execute(f'DELETE FROM "{PG_SCHEMA}".{table} WHERE namespace = $1', NAMESPACE)
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".namespaces WHERE name = $1', NAMESPACE)
    finally:
        await conn.close()


async def _fetchval(query: str, *args):
    conn = await asyncpg.connect(db_url())
    try:
        return await conn.fetchval(query, *args)
    finally:
        await conn.close()


@pytest.fixture
def namespace(monkeypatch):
    identity = auth.KeyIdentity(
        key_id=f"{KEY_PREFIX}key",
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


class FakeProvider:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.inputs: list[str] = []

    async def __call__(self, messages, schema, *, timeout):
        self.inputs.append(messages[-1]["content"])
        return self.replies.pop(0) if self.replies else {"notes": []}


def _run_next_job() -> job_store.JobBase:
    job = asyncio.run(job_store.claim_job("conversation", only_key_prefix=KEY_PREFIX))
    assert job is not None
    asyncio.run(job_store._run_claimed(job))
    return job


def _distilled_through(conversation_id: str) -> int:
    return asyncio.run(
        _fetchval(
            f'SELECT distilled_through FROM "{PG_SCHEMA}".conversation_sources WHERE id = $1',
            conversation_id,
        )
    )


def test_an_upload_is_distilled_into_linked_searchable_notes_once(namespace, monkeypatch):
    provider = FakeProvider(
        {
            "notes": [
                {
                    "content": NOTE_ARCH,
                    "kind": "note",
                    "turn_start": 0,
                    "turn_end": 1,
                    "tags": ["staging"],
                },
                {
                    "content": NOTE_DRAIN,
                    "kind": "decision",
                    "turn_start": 2,
                    "turn_end": 3,
                    "tags": ["deploy"],
                },
            ]
        }
    )
    monkeypatch.setattr(distill, "chat_json", provider)
    now = time.time()
    body = {
        "namespace": namespace,
        "origin": "claude_code",
        "external_session_id": "capture-it-session",
        "started_at": now - 600,
        "ended_at": now - 60,
        "turns": TURNS,
        "metadata": {"repo": "infra", "cwd": "/srv/infra"},
    }
    stored = client.post("/conversations", json=body)
    assert stored.status_code == 201, stored.json()
    conversation_id, job_id = stored.json()["id"], stored.json()["job_id"]
    assert job_id

    assert _run_next_job().job_id == job_id
    state = client.get(f"/conversations/jobs/{job_id}").json()
    assert state["status"] == "succeeded", state
    assert state["result"] == {"units": 2, "stored": 2, "refused": 0, "similar": 0}
    assert _distilled_through(conversation_id) == 4
    assert "[0] user:" in provider.inputs[0] and "[3] assistant:" in provider.inputs[0]

    response = client.post(
        "/search",
        json={
            "query": "How are staging images built and deployed?",
            "source": "memory",
            "namespaces": [namespace],
            "min_score": 0,
        },
    )
    assert response.status_code == 200, response.json()
    by_text = {hit["text"]: hit for hit in response.json()}
    assert {NOTE_ARCH, NOTE_DRAIN} <= set(by_text)
    arch, drain = by_text[NOTE_ARCH], by_text[NOTE_DRAIN]
    assert arch["conversation_id"] == drain["conversation_id"] == conversation_id
    assert (arch["turn_start"], arch["turn_end"]) == (0, 1)
    assert (drain["turn_start"], drain["turn_end"]) == (2, 3)
    assert sorted(arch["tags"]) == ["repo:infra", "staging"]
    assert drain["kind"] == "decision"

    extended = client.post(
        "/conversations", json={**body, "turns": [*TURNS, *MORE], "ended_at": now}
    )
    assert extended.status_code == 200, extended.json()
    assert extended.json()["turns"] == 6
    second_job = extended.json()["job_id"]
    assert second_job and second_job != job_id

    assert _run_next_job().job_id == second_job
    rendered = provider.inputs[1]
    assert "[4] user:" in rendered and "[5] assistant:" in rendered
    assert not any(f"[{index}]" in rendered for index in range(4))
    assert _distilled_through(conversation_id) == 6
    assert client.get(f"/conversations/jobs/{second_job}").json()["status"] == "succeeded"


def test_a_failed_job_admission_rolls_the_upload_back(namespace, monkeypatch):
    async def fail(**kwargs):
        raise RuntimeError("job admission failed")

    monkeypatch.setattr(job_store, "admit_conversation", fail)
    failing = TestClient(api.app, headers={"X-API-Key": "test-key"}, raise_server_exceptions=False)
    now = time.time()
    response = failing.post(
        "/conversations",
        json={
            "namespace": namespace,
            "origin": "claude_code",
            "external_session_id": "capture-it-rollback",
            "started_at": now - 60,
            "ended_at": now,
            "turns": TURNS,
        },
    )
    assert response.status_code == 500
    stored = asyncio.run(
        _fetchval(
            f'SELECT count(*) FROM "{PG_SCHEMA}".conversation_sources WHERE namespace = $1',
            namespace,
        )
    )
    assert stored == 0
