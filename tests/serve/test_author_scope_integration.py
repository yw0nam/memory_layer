"""Integration tests for the note read scope against the throwaway Postgres.

Seeds notes with real embeddings (one embed call, reused), then drives the real REST app
through `/search`, `GET /notes`, and `/save_memory`: the caller sees its own and shared
notes and never another author's private note, even when foreign notes fill the vector
candidate list; the near-duplicate check runs over that same scope. Every seeded row and
namespace is removed in ``finally``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, VllmEmbedder, db_url, embed_text
from memory_base.core.schema import ensure_schema
from memory_base.serve import api
from memory_base.serve.access import namespaces
from memory_base.serve.notes import store

pytestmark = pytest.mark.integration

A, B, C = "author-a", "author-b", "author-c"
# CANDIDATES_PER_SIGNAL is 50: more foreign notes than that would fill the candidate list.
FOREIGN_PRIVATE = 55


def _seed_row_sql() -> str:
    return f"""
    INSERT INTO "{PG_SCHEMA}".memory_chunks
      (id, source_type, source_ref, chunk_kind, session_id, content_raw,
       distilled, embedding, ts_last_active, namespace, metadata)
    VALUES ($1, 'agent_note', 'save_memory', 'work', $1, $2, $2, $3::halfvec, $4, $5, $6::jsonb)
    """


async def _drop(namespace: str) -> None:
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = $1', namespace
        )
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".namespaces WHERE name = $1', namespace)
    finally:
        await conn.close()


def _client(author: str) -> TestClient:
    return TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": author})


def test_search_never_returns_a_foreign_private_note():
    namespace = f"scope-search-{uuid.uuid4().hex[:8]}"
    marker = f"zzzscopesearch{uuid.uuid4().hex[:8]}"
    content = f"read scope integration pin {marker}: the deploy rule is unchanged"
    own_id = f"note:{namespace}:own"
    shared_id = f"note:{namespace}:shared"
    foreign_ids = [f"note:{namespace}:f{i}" for i in range(FOREIGN_PRIVATE)]

    async def seed() -> None:
        await namespaces.create_namespace(namespace)
        vector = await embed_text(VllmEmbedder(), content)
        now = time.time()
        conn = await asyncpg.connect(db_url())
        try:
            await ensure_schema(conn)
            rows = [
                (own_id, content, B, {}),
                (shared_id, content, A, {"shared": True}),
                *((i, content, A, {}) for i in foreign_ids),
            ]
            for row_id, text, author, extra in rows:
                await conn.execute(
                    _seed_row_sql(),
                    row_id,
                    text,
                    vector,
                    now,
                    namespace,
                    json.dumps({"author": author, **extra}),
                )
        finally:
            await conn.close()

    asyncio.run(seed())
    try:
        response = _client(B).post(
            "/search", json={"query": content, "source": "memory", "top_k": 20}
        )
        assert response.status_code == 200, response.text
        found = {hit["id"] for hit in response.json()}
        assert own_id in found
        assert shared_id in found
        assert not found & set(foreign_ids)
    finally:
        asyncio.run(db.close_pool())
        asyncio.run(_drop(namespace))


def test_a_shared_note_is_in_another_authors_default_reads():
    namespace = f"scope-shared-{uuid.uuid4().hex[:8]}"
    marker = f"zzzscopeshared{uuid.uuid4().hex[:8]}"
    content = f"read scope integration pin {marker}: the staging deploy window is Thursday"

    async def seed() -> str:
        await namespaces.create_namespace(namespace)
        saved = await store.save_note(
            content, tags=["test"], kind="work", namespace=namespace, author=A, shared=True
        )
        return saved["id"]

    note_id = asyncio.run(seed())
    try:
        listing = _client(B).get("/notes", params={"namespace": namespace})
        assert listing.status_code == 200
        assert note_id in [row["id"] for row in listing.json()]
        assert listing.json()[0]["shared"] is True

        found = _client(B).post(
            "/search", json={"query": content, "source": "memory", "namespaces": [namespace]}
        )
        assert found.status_code == 200
        assert note_id in [hit["id"] for hit in found.json()]
    finally:
        asyncio.run(db.close_pool())
        asyncio.run(_drop(namespace))


def test_the_near_duplicate_check_runs_over_the_writers_scope():
    namespace = f"scope-duplicate-{uuid.uuid4().hex[:8]}"
    marker = f"zzzscopedup{uuid.uuid4().hex[:8]}"
    private = f"read scope integration pin {marker}: the deploy rule stays the same"
    shared = f"read scope integration pin {marker}: the user drinks green tea every morning"

    async def seed() -> None:
        await namespaces.create_namespace(namespace)
        await store.save_note(private, tags=["test"], kind="work", namespace=namespace, author=A)
        await store.save_note(
            shared, tags=["test"], kind="work", namespace=namespace, author=C, shared=True
        )

    asyncio.run(seed())
    try:
        # A's private note is invisible: B may store the same text.
        accepted = _client(B).post(
            "/save_memory",
            json={"content": private, "kind": "work", "tags": ["test"], "namespace": namespace},
        )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["stored"] is True

        # C's shared note is in B's scope: the same text is refused as a near-duplicate.
        refused = _client(B).post(
            "/save_memory",
            json={"content": shared, "kind": "work", "tags": ["test"], "namespace": namespace},
        )
        assert refused.status_code == 409, refused.text
    finally:
        asyncio.run(db.close_pool())
        asyncio.run(_drop(namespace))


def test_a_hidden_foreign_supersede_beside_an_own_duplicate_is_refused():
    namespace = f"scope-supersede-{uuid.uuid4().hex[:8]}"
    marker = f"zzzscopesup{uuid.uuid4().hex[:8]}"
    own = f"read scope integration pin {marker}: the deploy runs before dawn"
    foreign = own
    rewrite = f"read scope integration pin {marker}: the deploy runs before dawn!"

    async def seed() -> str:
        await namespaces.create_namespace(namespace)
        await store.save_note(own, tags=["test"], kind="work", namespace=namespace, author=B)
        saved = await store.save_note(
            foreign, tags=["test"], kind="work", namespace=namespace, author=A
        )
        return saved["id"]

    foreign_id = asyncio.run(seed())
    try:
        response = _client(B).post(
            "/save_memory",
            json={
                "content": rewrite,
                "kind": "work",
                "tags": ["test"],
                "namespace": namespace,
                "supersedes": foreign_id,
            },
        )
        assert response.status_code == 409, response.text
        assert "in your scope" in response.json()["error"]
    finally:
        asyncio.run(db.close_pool())
        asyncio.run(_drop(namespace))
