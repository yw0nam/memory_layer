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
