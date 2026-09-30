"""ensure_schema converges a deployed memory_chunks table onto the current columns."""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.core.schema import ensure_schema

pytestmark = pytest.mark.integration


async def _memory_chunk_columns(conn: asyncpg.Connection) -> set[str]:
    rows = await conn.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = $1 AND table_name = 'memory_chunks'",
        PG_SCHEMA,
    )
    return {row["column_name"] for row in rows}


def test_ensure_schema_drops_the_idf_score_column():
    async def _run() -> tuple[set[str], set[str]]:
        conn = await asyncpg.connect(db_url())
        try:
            await conn.execute(
                f'ALTER TABLE "{PG_SCHEMA}".memory_chunks '
                "ADD COLUMN IF NOT EXISTS idf_score double precision"
            )
            await ensure_schema(conn)
            after_first = await _memory_chunk_columns(conn)
            await ensure_schema(conn)
            after_second = await _memory_chunk_columns(conn)
        finally:
            await conn.close()
        return after_first, after_second

    after_first, after_second = asyncio.run(_run())
    assert "idf_score" not in after_first
    assert after_second == after_first


def test_ensure_schema_admits_conversation_jobs_on_a_deployed_jobs_table():
    async def _run() -> str:
        conn = await asyncpg.connect(db_url())
        try:
            await conn.execute(
                f'ALTER TABLE "{PG_SCHEMA}".jobs DROP CONSTRAINT IF EXISTS jobs_kind_check'
            )
            await conn.execute(
                f'ALTER TABLE "{PG_SCHEMA}".jobs ADD CONSTRAINT jobs_kind_check '
                "CHECK (kind IN ('document', 'repo')) NOT VALID"
            )
            await ensure_schema(conn)
            await ensure_schema(conn)
            return await conn.fetchval(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'jobs_kind_check' AND conrelid = $1::regclass",
                f'"{PG_SCHEMA}".jobs',
            )
        finally:
            await conn.close()

    assert "'conversation'" in asyncio.run(_run())
