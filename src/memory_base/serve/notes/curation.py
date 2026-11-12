"""Database operations for memory lifecycle management."""

from __future__ import annotations

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once


async def notes_by_ids(ids: list[str]) -> list[dict]:
    """Return agent notes matching the supplied identifiers."""
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, content_raw AS text, ts_last_active, archived_at
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note' AND id = ANY($1::text[])
            ORDER BY ts_last_active ASC
            """,
            ids,
        )
        return [dict(row) for row in rows]


async def delete_notes(ids: list[str]) -> int:
    """Delete agent notes matching the supplied identifiers."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        status = await conn.execute(
            f"""
            DELETE FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note' AND id = ANY($1::text[])
            """,
            ids,
        )
        return int(status.rsplit(" ", 1)[-1])


async def archive_rows(ids: list[str], now: float, archived_by: str) -> int:
    """Archive active agent notes matching the supplied identifiers, stamping the archiver."""
    async with db.acquire() as conn:
        status = await conn.execute(
            f"""
            UPDATE "{PG_SCHEMA}".memory_chunks
            SET archived_at = $2,
                metadata = metadata || jsonb_build_object('archived_by', $3::text)
            WHERE source_type = 'agent_note' AND id = ANY($1::text[]) AND archived_at IS NULL
            """,
            ids,
            now,
            archived_by,
        )
        return int(status.rsplit(" ", 1)[-1])


async def rows_by_ids(ids: list[str], notes_only: bool = False) -> list[dict]:
    """Return lifecycle fields for rows matching the supplied identifiers.

    ``notes_only`` drops every row that is not an agent note.
    """
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, chunk_kind AS kind, archived_at,
                   metadata->>'archived_by' AS archived_by
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE id = ANY($1::text[])
              AND (NOT $2::boolean OR source_type = 'agent_note')
            ORDER BY id
            """,
            ids,
            notes_only,
        )
        return [dict(row) for row in rows]
