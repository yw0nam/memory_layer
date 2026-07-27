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
    DB_URL,
    EMB_DIM,
    PG_SCHEMA,
    VllmEmbedder,
    embed_text,
    vector_literal,
)
from memory_base.core.schema import ensure_schema
from memory_base.serve import api

client = TestClient(api.app)


def _db_reachable() -> bool:
    async def _check() -> None:
        conn = await asyncpg.connect(DB_URL, timeout=5)
        await conn.close()

    try:
        asyncio.run(_check())
        return True
    except Exception:
        return False


_DB = _db_reachable()
requires_db = pytest.mark.skipif(not _DB, reason=f"DB not reachable at {DB_URL}")


def _random_vec(seed: int) -> str:
    return vector_literal(np.random.default_rng(seed).normal(size=EMB_DIM))


def _near_dup_vecs(seed: int) -> tuple[str, str]:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=EMB_DIM)
    noisy = base + rng.normal(scale=0.01, size=EMB_DIM)
    return vector_literal(base), vector_literal(noisy)


async def _seed_row(
    conn: asyncpg.Connection,
    row_id: str,
    content: str,
    embedding_lit: str,
    ts_last_active: float,
    last_hit_at: float | None = None,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO "{PG_SCHEMA}".memory_chunks
          (id, source_type, source_ref, chunk_kind, session_id, content_raw,
           distilled, embedding, ts_last_active, idf_score, metadata,
           hit_count, last_hit_at)
        VALUES ($1,'agent_note','save_memory','note',$1,$2,$2,$3::halfvec,$4,NULL,
                '{{}}'::jsonb, 0, $5)
        ON CONFLICT (id) DO NOTHING
        """,
        row_id,
        content,
        embedding_lit,
        ts_last_active,
        last_hit_at,
    )


async def _delete_rows(ids: list[str]) -> None:
    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE id = ANY($1::text[])', ids
        )
    finally:
        await conn.close()


async def _row_count(row_id: str) -> int:
    conn = await asyncpg.connect(DB_URL)
    try:
        return await conn.fetchval(
            f'SELECT count(*) FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', row_id
        )
    finally:
        await conn.close()


async def _archived_at(row_id: str):
    conn = await asyncpg.connect(DB_URL)
    try:
        return await conn.fetchval(
            f'SELECT archived_at FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', row_id
        )
    finally:
        await conn.close()


@pytest.mark.integration
@requires_db
def test_full_archive_and_note_lifecycle():
    now_real = time.time()
    token = f"zzzarchiveflow{int(now_real)}"
    old_id = f"archiveflow-old-{int(now_real)}"
    dup_a_id = f"archiveflow-dupa-{int(now_real)}"
    dup_b_id = f"archiveflow-dupb-{int(now_real)}"

    # Natural prose, not keyword soup: the reranker scores answer-like text and
    # buries non-informative token strings even on a verbatim match.
    content_old = (
        f"The {token} cold tier experiment settled on archiving rows older than 180 days "
        f"that also went 90 days without a retrieval hit, keeping the thresholds conservative."
    )
    # The token stays out of the dup rows: the archive assertions below treat it
    # as a marker unique to the archived row, and the dup pair is asserted by id.
    content_dup_a = "near duplicate pair alpha rendition of a troubleshooting note"
    content_dup_b = "near duplicate pair beta rendition of a troubleshooting note"

    old_ts = now_real - 200 * 86400  # well past COLD_AGE_DAYS=180 and COLD_UNHIT_DAYS=90
    dup_a_vec, dup_b_vec = _near_dup_vecs(seed=7)

    async def _seed_all() -> None:
        # The archived row gets a real embedding of its content: the archive /
        # restore assertions go through full retrieval, where a random vector
        # would sink below the RRF candidate cut regardless of archived state.
        old_vec = await embed_text(VllmEmbedder(), content_old)
        conn = await asyncpg.connect(DB_URL)
        try:
            await ensure_schema(conn)
            await _seed_row(conn, old_id, content_old, old_vec, old_ts, None)
            await _seed_row(conn, dup_a_id, content_dup_a, dup_a_vec, now_real, None)
            await _seed_row(conn, dup_b_id, content_dup_b, dup_b_vec, now_real, None)
        finally:
            await conn.close()

    asyncio.run(_delete_rows([old_id, dup_a_id, dup_b_id]))
    asyncio.run(_seed_all())
    try:
        # ---- /admin/archive: dry-run lists our aged row, confirm archives it ----
        dry = client.post("/admin/archive", json={})
        assert dry.status_code == 200
        candidates = dry.json()["candidates"]
        candidate_ids = {c["id"] for c in candidates}
        assert old_id in candidate_ids
        assert dup_a_id not in candidate_ids
        assert dup_b_id not in candidate_ids
        (matching,) = [c for c in candidates if c["id"] == old_id]
        assert "hit_count" in matching and "last_hit_at" in matching

        confirm = client.post("/admin/archive", json={"confirm": True})
        assert confirm.status_code == 200
        assert confirm.json()["archived"] >= 1
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
        # Archived rows may be superseded; the caller must be able to tell them apart.
        assert all(h.get("archived") is True for h in archived_hits)
        assert all("archived" not in h for h in archived_search.json() if token not in h["text"])

        # ---- /admin/restore: dry-run then confirm brings it back to default search ----
        restore_dry = client.post("/admin/restore", json={"ids": [old_id]})
        assert restore_dry.status_code == 200
        assert any(r["id"] == old_id for r in restore_dry.json()["rows"])

        restore_confirm = client.post("/admin/restore", json={"ids": [old_id], "confirm": True})
        assert restore_confirm.status_code == 200
        assert restore_confirm.json() == {"restored": 1}
        assert asyncio.run(_archived_at(old_id)) is None
        # Restore returns the row to the active pool; default-search RANKING is
        # not asserted because time decay legitimately buries a 200-day-old row.
        # Active-pool membership is verified via /admin/notes below (it filters
        # on archived_at IS NULL).

        # ---- /admin/notes lists the aged seeded note ----
        notes = client.get("/admin/notes", params={"older_than_days": 100})
        assert notes.status_code == 200
        assert any(row["id"] == old_id for row in notes.json())

        # ---- /admin/notes/delete: dry-run then confirm removes it ----
        delete_dry = client.post("/admin/notes/delete", json={"ids": [old_id]})
        assert delete_dry.status_code == 200
        assert any(row["id"] == old_id for row in delete_dry.json()["rows"])

        delete_confirm = client.post("/admin/notes/delete", json={"ids": [old_id], "confirm": True})
        assert delete_confirm.status_code == 200
        assert delete_confirm.json() == {"deleted": 1}
        assert asyncio.run(_row_count(old_id)) == 0

        # ---- /admin/duplicates finds the seeded near-dup pair ----
        duplicates = client.get("/admin/duplicates", params={"threshold": 0.95, "limit": 50})
        assert duplicates.status_code == 200
        pairs = duplicates.json()["pairs"]
        assert any({p["a"]["id"], p["b"]["id"]} == {dup_a_id, dup_b_id} for p in pairs)
    finally:
        asyncio.run(_delete_rows([old_id, dup_a_id, dup_b_id]))
