"""Conversation sources end to end against the session's throwaway Postgres.

Stores a source, links two notes to its turns, and reads both lanes back
through the real REST app: search returns the notes with their link, the
source slice returns exactly the linked turn, and no search ever returns the
source's own text. The embedder and reranker are the configured live endpoints.
"""

from __future__ import annotations

import asyncio
import time

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api, namespaces

pytestmark = pytest.mark.integration

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

NAMESPACE = "conv-source-it"
TURNS = [
    {
        "role": "user",
        "text": "The staging cluster runs on three ARM nodes behind one HAProxy instance.",
    },
    {
        "role": "assistant",
        "text": "Then every container image for staging has to be built for linux/arm64 too.",
    },
    {
        "role": "user",
        "text": "Drain the HAProxy backend before each staging deploy or connections drop.",
    },
]
NOTE_ARCH = "Staging runs on ARM nodes, so every staging container image is built multi-arch."
NOTE_DRAIN = "A staging deploy drains the HAProxy backend first; skipping it drops connections."


async def _cleanup() -> None:
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = $1', NAMESPACE
        )
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".conversation_sources WHERE namespace = $1', NAMESPACE
        )
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".namespaces WHERE name = $1', NAMESPACE)
    finally:
        await conn.close()


@pytest.fixture
def namespace():
    asyncio.run(_cleanup())
    asyncio.run(namespaces.create_namespace(NAMESPACE))
    yield NAMESPACE
    asyncio.run(_cleanup())


def _save(content: str, conversation_id: str, turn_start: int, turn_end: int) -> dict:
    response = client.post(
        "/save_memory",
        json={
            "namespace": NAMESPACE,
            "author": "claude-code",
            "content": content,
            "tags": ["staging"],
            "conversation_id": conversation_id,
            "turn_start": turn_start,
            "turn_end": turn_end,
        },
    )
    assert response.status_code == 200, response.json()
    return response.json()


def _search(query: str) -> list[dict]:
    response = client.post(
        "/search",
        json={"query": query, "source": "memory", "namespaces": [NAMESPACE], "min_score": 0},
    )
    assert response.status_code == 200, response.json()
    return response.json()


def test_linked_notes_carry_their_source_and_the_source_never_surfaces(namespace):
    now = time.time()
    source = {
        "namespace": namespace,
        "origin": "claude_code",
        "external_session_id": "integration-session-1",
        "started_at": now - 600,
        "ended_at": now - 60,
        "turns": TURNS,
    }
    stored = client.post("/conversations", json=source)
    assert stored.status_code == 201, stored.json()
    assert stored.json()["created"] is True
    assert stored.json()["turns"] == 3
    conversation_id = stored.json()["id"]

    arch = _save(NOTE_ARCH, conversation_id, 0, 1)
    drain = _save(NOTE_DRAIN, conversation_id, 2, 2)
    assert arch["stored"] is True and drain["stored"] is True
    assert arch["id"].startswith(f"note:{namespace}:")

    hits = _search("How are staging container images built and deployed?")
    by_id = {hit["id"]: hit for hit in hits}
    assert {arch["id"], drain["id"]} <= set(by_id)
    assert by_id[arch["id"]]["conversation_id"] == conversation_id
    assert (by_id[arch["id"]]["turn_start"], by_id[arch["id"]]["turn_end"]) == (0, 1)
    assert (by_id[drain["id"]]["turn_start"], by_id[drain["id"]]["turn_end"]) == (2, 2)
    assert by_id[drain["id"]]["kind"] == "note"
    assert by_id[drain["id"]]["tags"] == ["staging"]

    sliced = client.get(f"/conversations/{conversation_id}?turn_start=2&turn_end=2")
    assert sliced.status_code == 200
    assert sliced.json()["turns"] == [{"index": 2, **TURNS[2]}]

    turn_texts = {turn["text"] for turn in TURNS}
    copied = _search(TURNS[2]["text"])
    assert drain["id"] in {hit["id"] for hit in copied}
    for hit in copied:
        assert hit["text"] not in turn_texts
        assert hit["id"] in {arch["id"], drain["id"]}

    replayed = client.post("/conversations", json=source)
    assert replayed.status_code == 200
    assert replayed.json() == {"id": conversation_id, "created": False, "turns": 3}
    rewritten = client.post(
        "/conversations",
        json={**source, "turns": [*TURNS, {"role": "assistant", "text": "A new turn."}]},
    )
    assert rewritten.status_code == 409


def test_a_note_linked_to_a_missing_source_is_refused(namespace):
    response = client.post(
        "/save_memory",
        json={
            "namespace": namespace,
            "author": "claude-code",
            "content": NOTE_ARCH,
            "tags": ["staging"],
            "conversation_id": "conv:0000000000000000",
            "turn_start": 0,
            "turn_end": 0,
        },
    )
    assert response.status_code == 400
    assert "unknown conversation_id" in response.json()["error"]
