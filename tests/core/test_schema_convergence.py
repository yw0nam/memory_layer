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

    async def _run() -> tuple[int, str, set[str], set[str], set[str]]:
        conn = await asyncpg.connect(db_url())
        try:
            for statement in (
                "ALTER TABLE {s}.jobs ADD COLUMN IF NOT EXISTS conversation_id text",
                "ALTER TABLE {s}.jobs ADD COLUMN IF NOT EXISTS result jsonb",
                "ALTER TABLE {s}.jobs DROP CONSTRAINT IF EXISTS jobs_kind_check",
                "ALTER TABLE {s}.jobs ADD CONSTRAINT jobs_kind_check "
                "CHECK (kind IN ('document', 'repo', 'conversation')) NOT VALID",
                "ALTER TABLE {s}.jobs DROP CONSTRAINT IF EXISTS jobs_conversation_check",
                "ALTER TABLE {s}.jobs ADD CONSTRAINT jobs_conversation_check "
                "CHECK (kind <> 'conversation' OR "
                "(namespace IS NOT NULL AND conversation_id IS NOT NULL))",
                "CREATE INDEX IF NOT EXISTS jobs__conversation_active "
                "ON {s}.jobs (conversation_id, status) WHERE kind = 'conversation'",
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
            constraints = {
                row["conname"]
                for row in await conn.fetch(
                    "SELECT conname FROM pg_constraint WHERE conrelid = $1::regclass",
                    f'"{PG_SCHEMA}".jobs',
                )
            }
            indexes = {
                row["indexname"]
                for row in await conn.fetch(
                    "SELECT indexname FROM pg_indexes WHERE schemaname = $1", PG_SCHEMA
                )
            }
        finally:
            await conn.close()
        return remaining, kind_check, jobs, constraints, indexes

    remaining, kind_check, jobs, constraints, indexes = asyncio.run(_run())
    assert remaining == 0
    assert "'conversation'" not in kind_check
    assert "'document'" in kind_check and "'repo'" in kind_check
    assert not {"conversation_id", "result"} & jobs
    assert "jobs_conversation_check" not in constraints
    assert "jobs__conversation_active" not in indexes


LEGACY_LINK_COLUMNS = {"conversation_id", "source_turn_start", "source_turn_end"}


async def _doc_rows_state(conn: asyncpg.Connection) -> tuple[bool, bool, list[str]]:
    flags = await conn.fetchrow(
        "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid = $1::regclass",
        f'"{PG_SCHEMA}".doc_rows',
    )
    policies = await conn.fetch(
        "SELECT policyname FROM pg_policies WHERE schemaname = $1 AND tablename = 'doc_rows' "
        "ORDER BY policyname",
        PG_SCHEMA,
    )
    return (
        flags["relrowsecurity"],
        flags["relforcerowsecurity"],
        [row["policyname"] for row in policies],
    )


async def _final_state(conn: asyncpg.Connection) -> dict[str, object]:
    tables = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = $1 ORDER BY tablename", PG_SCHEMA
    )
    indexes = await conn.fetch(
        "SELECT indexname FROM pg_indexes WHERE schemaname = $1 ORDER BY indexname", PG_SCHEMA
    )
    constraints = await conn.fetch(
        "SELECT conname, pg_get_constraintdef(oid) AS def FROM pg_constraint "
        "WHERE conrelid = $1::regclass ORDER BY conname",
        f'"{PG_SCHEMA}".jobs',
    )
    return {
        "tables": [row["tablename"] for row in tables],
        "indexes": [row["indexname"] for row in indexes],
        "chunk_columns": sorted(await _memory_chunk_columns(conn)),
        "jobs_constraints": [(row["conname"], row["def"]) for row in constraints],
        "doc_rows": await _doc_rows_state(conn),
    }


def test_ensure_schema_drops_the_legacy_conversation_sources_and_note_links():
    async def _run() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        conn = await asyncpg.connect(db_url())
        try:
            for statement in (
                "ALTER TABLE {s}.memory_chunks ADD COLUMN IF NOT EXISTS conversation_id text",
                "ALTER TABLE {s}.memory_chunks ADD COLUMN IF NOT EXISTS source_turn_start int",
                "ALTER TABLE {s}.memory_chunks ADD COLUMN IF NOT EXISTS source_turn_end int",
                "CREATE INDEX IF NOT EXISTS memory_chunks__conversation "
                "ON {s}.memory_chunks (conversation_id)",
                "CREATE TABLE IF NOT EXISTS {s}.conversation_sources "
                "(id text PRIMARY KEY, turns jsonb NOT NULL)",
                "CREATE INDEX IF NOT EXISTS conversation_sources__started "
                "ON {s}.conversation_sources (id)",
            ):
                await conn.execute(statement.format(s=f'"{PG_SCHEMA}"'))
            seeded = await _final_state(conn)
            await ensure_schema(conn)
            after_first = await _final_state(conn)
            await ensure_schema(conn)
            after_second = await _final_state(conn)
        finally:
            await conn.close()
        return seeded, after_first, after_second

    seeded, after_first, after_second = asyncio.run(_run())
    assert "conversation_sources" in seeded["tables"]
    assert LEGACY_LINK_COLUMNS <= set(seeded["chunk_columns"])
    assert "memory_chunks__conversation" in seeded["indexes"]
    assert "conversation_sources" not in after_first["tables"]
    assert not LEGACY_LINK_COLUMNS & set(after_first["chunk_columns"])
    assert "occurred_at" in after_first["chunk_columns"]
    assert "memory_chunks__conversation" not in after_first["indexes"]
    assert "conversation_sources__started" not in after_first["indexes"]
    assert after_first["doc_rows"][:2] == (True, True)
    assert after_first["doc_rows"][2]
    assert after_first["jobs_constraints"]
    assert after_second == after_first
