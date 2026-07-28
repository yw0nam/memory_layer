"""Integration tests for access logging.

Exercises the real REST app against Postgres + vLLM (embedder/reranker): a
POST /search must write one retrieval_log row and bump hit_count/last_hit_at
on every returned memory_chunks hit; POST /save_memory must roundtrip through
the real DB; ensure_schema must be idempotent and add the access-log columns
and table. Skipped when the DB is unreachable, matching the pattern in
tests/test_save_memory.py.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from starlette.testclient import TestClient

import asyncpg

from memory_base.core.config import DB_URL, PG_SCHEMA
from memory_base.core.schema import ensure_schema
from memory_base.serve import api
from memory_base.serve.notes import build_note_row, save_note

NOW = 1_700_000_000.0


@pytest.fixture()
def client():
    with TestClient(api.app) as c:
        yield c


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


async def _delete_note(note_id: str) -> None:
    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', note_id)
    finally:
        await conn.close()


async def _fetch_chunk(note_id: str):
    conn = await asyncpg.connect(DB_URL)
    try:
        return await conn.fetchrow(
            f'SELECT id, hit_count, last_hit_at FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1',
            note_id,
        )
    finally:
        await conn.close()


async def _fetch_latest_retrieval_log(query_text: str, source: str):
    conn = await asyncpg.connect(DB_URL)
    try:
        return await conn.fetchrow(
            f'SELECT * FROM "{PG_SCHEMA}".retrieval_log WHERE query=$1 AND source=$2 '
            f"ORDER BY id DESC LIMIT 1",
            query_text,
            source,
        )
    finally:
        await conn.close()


async def _delete_retrieval_log(query_text: str) -> None:
    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".retrieval_log WHERE query=$1', query_text)
    finally:
        await conn.close()


# ---- POST /search logs retrieval_log + bumps hit columns ------------------


async def _fetch_chunks_by_ids(ids: list[str]):
    conn = await asyncpg.connect(DB_URL)
    try:
        return await conn.fetch(
            f'SELECT id, hit_count, last_hit_at FROM "{PG_SCHEMA}".memory_chunks '
            "WHERE id = ANY($1::text[])",
            ids,
        )
    finally:
        await conn.close()


@pytest.mark.integration
@requires_db
def test_search_logs_retrieval_and_bumps_hit_columns(client):
    """Every returned hit is logged and its memory_chunks row is bumped.

    Ranking is not asserted (which rows come back depends on the corpus);
    the pin is the mechanism: one retrieval_log row per search, and every
    returned memory_chunks id has hit_count >= 1 with last_hit_at set to
    the moment of this search.
    """
    content = "access-log integration pin: zzzaccesslogpin unique retrieval marker"
    note_id = build_note_row(content, "note", None, NOW)["id"]
    asyncio.run(_delete_note(note_id))
    asyncio.run(_delete_retrieval_log(content))
    try:
        asyncio.run(save_note(content))
        t0 = time.time()

        response = client.post("/search", json={"query": content, "source": "memory", "top_k": 5})
        assert response.status_code == 200
        assert response.json()

        log_row = asyncio.run(_fetch_latest_retrieval_log(content, "memory"))
        assert log_row is not None
        assert log_row["query"] == content
        assert log_row["source"] == "memory"
        assert log_row["hit_ids"]
        assert log_row["ts"] >= t0

        bumped = asyncio.run(_fetch_chunks_by_ids(list(log_row["hit_ids"])))
        assert bumped  # history hits must resolve to memory_chunks rows
        for row in bumped:
            assert row["hit_count"] >= 1
            assert row["last_hit_at"] is not None and row["last_hit_at"] >= t0
    finally:
        asyncio.run(_delete_note(note_id))
        asyncio.run(_delete_retrieval_log(content))


# ---- POST /save_memory roundtrip -------------------------------------------


@pytest.mark.integration
@requires_db
def test_save_memory_endpoint_roundtrip_and_dedup(client):
    content = "access-log integration pin: save_memory REST endpoint roundtrip dedup check"
    note_id = build_note_row(content, "note", None, NOW)["id"]
    asyncio.run(_delete_note(note_id))
    try:
        first = client.post("/save_memory", json={"content": content})
        assert first.status_code == 200
        assert first.json() == {
            "id": note_id,
            "kind": "note",
            "stored": True,
            "superseded": None,
            "similar": [],
        }

        second = client.post("/save_memory", json={"content": content})
        assert second.status_code == 200
        assert second.json()["id"] == note_id
        assert second.json()["stored"] is False
        assert second.json()["superseded"] is None
        assert second.json()["similar"] == []

        row = asyncio.run(_fetch_chunk(note_id))
        assert row is not None
        assert row["id"] == note_id
    finally:
        asyncio.run(_delete_note(note_id))


# ---- schema idempotency -----------------------------------------------------


@pytest.mark.integration
@requires_db
def test_ensure_schema_idempotent_and_adds_access_log_objects():
    async def _run():
        conn = await asyncpg.connect(DB_URL)
        try:
            await ensure_schema(conn)
            await ensure_schema(conn)  # second call must not raise
            table_exists = await conn.fetchval(
                "SELECT to_regclass($1) IS NOT NULL", f"{PG_SCHEMA}.retrieval_log"
            )
            columns = await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=$1 AND table_name='memory_chunks'",
                PG_SCHEMA,
            )
        finally:
            await conn.close()
        return table_exists, {row["column_name"] for row in columns}

    table_exists, column_names = asyncio.run(_run())
    assert table_exists is True
    assert {"last_hit_at", "hit_count"} <= column_names
