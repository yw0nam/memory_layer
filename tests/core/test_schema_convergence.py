"""ensure_schema converges deployed tables onto the current columns and constraints."""

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


async def _columns(conn: asyncpg.Connection, table: str) -> set[str]:
    rows = await conn.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = $1 AND table_name = $2",
        PG_SCHEMA,
        table,
    )
    return {row["column_name"] for row in rows}


def test_ensure_schema_drops_conversation_jobs_from_a_deployed_jobs_table():
    job_id = "it-schema-convergence-conversation"

    async def _run() -> tuple[int, str, set[str], set[str]]:
        conn = await asyncpg.connect(db_url())
        try:
            for statement in (
                "ALTER TABLE {s}.jobs ADD COLUMN IF NOT EXISTS conversation_id text",
                "ALTER TABLE {s}.jobs ADD COLUMN IF NOT EXISTS result jsonb",
                "ALTER TABLE {s}.conversation_sources "
                "ADD COLUMN IF NOT EXISTS distilled_through int NOT NULL DEFAULT 0",
                "ALTER TABLE {s}.jobs DROP CONSTRAINT IF EXISTS jobs_kind_check",
                "ALTER TABLE {s}.jobs ADD CONSTRAINT jobs_kind_check "
                "CHECK (kind IN ('document', 'repo', 'conversation')) NOT VALID",
            ):
                await conn.execute(statement.format(s=f'"{PG_SCHEMA}"'))
            await conn.execute(
                f"""INSERT INTO "{PG_SCHEMA}".jobs
                (job_id, kind, status, key_id, key_label, namespace, conversation_id)
                VALUES ($1, 'conversation', 'queued', $1, 'test', 'default', 'conv:0')""",
                job_id,
            )
            await ensure_schema(conn)
            await ensure_schema(conn)
            remaining = await conn.fetchval(
                f'SELECT count(*) FROM "{PG_SCHEMA}".jobs WHERE job_id = $1', job_id
            )
            kind_check = await conn.fetchval(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'jobs_kind_check' AND conrelid = $1::regclass",
                f'"{PG_SCHEMA}".jobs',
            )
            jobs = await _columns(conn, "jobs")
            sources = await _columns(conn, "conversation_sources")
        finally:
            await conn.close()
        return remaining, kind_check, jobs, sources

    remaining, kind_check, jobs, sources = asyncio.run(_run())
    assert remaining == 0
    assert "'conversation'" not in kind_check
    assert "'document'" in kind_check and "'repo'" in kind_check
    assert not {"conversation_id", "result"} & jobs
    assert "distilled_through" not in sources
