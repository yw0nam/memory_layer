"""Integration test for the note archive and delete flow (real DB).

Seeds synthetic rows directly via asyncpg, then drives the real REST app through:

    /admin/archive (dry-run -> confirm) -> /search default vs include_archived
    -> /admin/notes/delete (dry-run -> confirm)

Every seeded row is deleted in ``finally`` regardless of outcome.
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
           distilled, embedding, ts_last_active, metadata)
        VALUES ($1,$6,'save_memory',$5,$1,$2,$2,$3::halfvec,$4,
                jsonb_build_object('author', $7::text))
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
        default_search = client.post("/search", json={"query": content_old, "top_k": 20})
        assert default_search.status_code == 200
        assert not any(token in h["text"] for h in default_search.json())

        archived_search = client.post(
            "/search",
            json={
                "query": content_old,
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
