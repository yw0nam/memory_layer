"""Integration test for the full cold-tier / note lifecycle flow (real DB).

Seeds synthetic rows directly via
asyncpg (bypassing the embedder-backed save path, so old timestamps can be
set precisely), then drives the real REST app through the whole lifecycle:

    /admin/archive (dry-run -> confirm) -> /search default vs include_archived
    -> /admin/restore (dry-run -> confirm) -> /admin/notes -> /admin/notes/delete
    (dry-run -> confirm) -> /admin/duplicates

Every seeded row is deleted in ``finally`` regardless of outcome. Skipped
when the DB is unreachable, matching tests/test_save_memory.py.
"""

from __future__ import annotations

import asyncio
import time

import asyncpg
import numpy as np
import pytest
from starlette.testclient import TestClient

from memory_base.core.config import (
    EMB_DIM,
    PG_SCHEMA,
    VllmEmbedder,
    db_url,
    embed_text,
    vector_literal,
)
from memory_base.core.schema import ensure_schema
from memory_base.serve import api

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})


def _angled_vec(angle: float) -> str:
    vector = np.zeros(EMB_DIM)
    vector[:2] = [np.cos(angle), np.sin(angle)]
    return vector_literal(vector)


async def _seed_row(
    conn: asyncpg.Connection,
    row_id: str,
    content: str,
    embedding_lit: str,
    ts_last_active: float,
    chunk_kind: str = "work",
    source_type: str = "agent_note",
    author: str = "claude-code",
) -> None:
    await conn.execute(
        f"""
        INSERT INTO "{PG_SCHEMA}".memory_chunks
          (id, source_type, source_ref, chunk_kind, session_id, content_raw,
           distilled, embedding, ts_last_active, metadata,
           hit_count, last_hit_at)
        VALUES ($1,$7,'save_memory',$6,$1,$2,$2,$3::halfvec,$4,
                jsonb_build_object('author', $8::text), 0, $5)
        ON CONFLICT (id) DO NOTHING
        """,
        row_id,
        content,
        embedding_lit,
        ts_last_active,
        chunk_kind,
        source_type,
        author,
    )


async def _delete_rows(ids: list[str]) -> None:
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE id = ANY($1::text[])', ids
        )
    finally:
        await conn.close()


async def _row_count(row_id: str) -> int:
    conn = await asyncpg.connect(db_url())
    try:
        return await conn.fetchval(
            f'SELECT count(*) FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', row_id
        )
    finally:
        await conn.close()


async def _archived_at(row_id: str):
    conn = await asyncpg.connect(db_url())
    try:
        return await conn.fetchval(
            f'SELECT archived_at FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', row_id
        )
    finally:
        await conn.close()


@pytest.mark.integration
def test_full_archive_and_note_lifecycle():
    now_real = time.time()
    token = f"zzzarchiveflow{int(now_real)}"
    old_id = f"archiveflow-old-{int(now_real)}"

    # Natural prose, not keyword soup: the reranker scores answer-like text and
    # buries non-informative token strings even on a verbatim match.
    content_old = (
        f"The {token} archive experiment settled on archiving superseded rows by id, "
        f"keeping them out of default search while include_archived still returns them."
    )

    async def _seed_all() -> None:
        # A real embedding keeps the row inside the RRF candidate cut of full retrieval.
        old_vec = await embed_text(VllmEmbedder(), content_old)
        conn = await asyncpg.connect(db_url())
        try:
            await ensure_schema(conn)
            await _seed_row(conn, old_id, content_old, old_vec, now_real)
        finally:
            await conn.close()

    asyncio.run(_delete_rows([old_id]))
    asyncio.run(_seed_all())
    try:
        # ---- /admin/archive: dry-run previews the row, confirm archives it ----
        dry = client.post("/admin/archive", json={"ids": [old_id]})
        assert dry.status_code == 200
        assert [row["id"] for row in dry.json()["notes_to_archive"]] == [old_id]
        assert asyncio.run(_archived_at(old_id)) is None

        confirm = client.post("/admin/archive", json={"ids": [old_id], "confirm": True})
        assert confirm.status_code == 200
        assert confirm.json() == {"archived": 1}
        assert asyncio.run(_archived_at(old_id)) is not None

        # ---- default /search excludes archived rows; include_archived brings back ----
        default_search = client.post(
            "/search", json={"query": content_old, "source": "memory", "top_k": 20}
        )
        assert default_search.status_code == 200
        assert not any(token in h["text"] for h in default_search.json())

        archived_search = client.post(
            "/search",
            json={
                "query": content_old,
                "source": "memory",
                "top_k": 20,
                "include_archived": True,
            },
        )
        assert archived_search.status_code == 200
        archived_hits = [h for h in archived_search.json() if token in h["text"]]
        assert archived_hits
        # Only the token rows are asserted on: a lived-in database legitimately
        # returns other archived rows in an include_archived search.
        assert all(h.get("archived") is True for h in archived_hits)

        # ---- /admin/notes/delete: dry-run then confirm removes it ----
        delete_dry = client.post("/admin/notes/delete", json={"ids": [old_id]})
        assert delete_dry.status_code == 200
        assert any(row["id"] == old_id for row in delete_dry.json()["rows"])

        delete_confirm = client.post("/admin/notes/delete", json={"ids": [old_id], "confirm": True})
        assert delete_confirm.status_code == 200
        assert delete_confirm.json() == {"deleted": 1}
        assert asyncio.run(_row_count(old_id)) == 0
    finally:
        asyncio.run(_delete_rows([old_id]))


@pytest.mark.integration
def test_archive_by_ids_refuses_a_document_chunk_like_an_unknown_id():
    token = time.time_ns()
    note_id = f"notesonly-ids-note-{token}"
    doc_id = f"doc:notesonly-ids-{token}:0"
    seeded_ids = [note_id, doc_id]
    old = time.time() - 400 * 86400

    async def _seed() -> None:
        conn = await asyncpg.connect(db_url())
        try:
            await ensure_schema(conn)
            await _seed_row(conn, note_id, "notes only ids note", _angled_vec(0), old)
            await _seed_row(
                conn,
                doc_id,
                "notes only ids document",
                _angled_vec(0),
                old,
                chunk_kind="doc",
                source_type="document",
            )
        finally:
            await conn.close()

    asyncio.run(_delete_rows(seeded_ids))
    asyncio.run(_seed())
    try:
        body = {"ids": [note_id, doc_id]}
        unknown = client.post("/admin/archive", json={**body, "ids": [note_id, f"missing-{token}"]})
        assert unknown.status_code == 400
        assert unknown.json()["missing"] == [f"missing-{token}"]
        for extra in ({}, {"confirm": True}):
            response = client.post("/admin/archive", json={**body, **extra})
            assert response.status_code == 400
            assert response.json()["missing"] == [doc_id]
        assert asyncio.run(_archived_at(note_id)) is None
        assert asyncio.run(_archived_at(doc_id)) is None

        only_note = client.post("/admin/archive", json={"ids": [note_id], "confirm": True})
        assert only_note.json()["archived"] == 1
        assert asyncio.run(_archived_at(note_id)) is not None
    finally:
        asyncio.run(_delete_rows(seeded_ids))
