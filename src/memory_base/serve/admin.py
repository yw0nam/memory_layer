"""Database operations for memory lifecycle management."""

from __future__ import annotations

import os

import asyncpg

from memory_base.common import DB_URL, PG_SCHEMA

COLD_AGE_DAYS = int(os.getenv("COLD_AGE_DAYS", "180"))
COLD_UNHIT_DAYS = int(os.getenv("COLD_UNHIT_DAYS", "90"))
DAY_SECONDS = 86400.0
TEXT_LIMIT = 2000


def is_cold(
    ts_last_active: float,
    last_hit_at: float | None,
    now: float,
    age_days: int,
    unhit_days: int,
) -> bool:
    """Return whether a row is older than both cold-tier cutoffs."""
    effective_hit_at = last_hit_at if last_hit_at is not None else ts_last_active
    return (
        ts_last_active < now - age_days * DAY_SECONDS
        and effective_hit_at < now - unhit_days * DAY_SECONDS
    )


async def list_old_notes(older_than_days: int) -> list[dict]:
    """Return active agent notes older than the requested age."""
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch(
            f"""
            SELECT id, content_raw AS text, ts_last_active, hit_count, last_hit_at
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note'
              AND archived_at IS NULL
              AND ts_last_active
                  < extract(epoch FROM now()) - $1::double precision * {DAY_SECONDS}
            ORDER BY ts_last_active ASC
            """,
            older_than_days,
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def notes_by_ids(ids: list[str]) -> list[dict]:
    """Return agent notes matching the supplied identifiers."""
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch(
            f"""
            SELECT id, content_raw AS text, ts_last_active, hit_count, last_hit_at,
                   archived_at
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note' AND id = ANY($1::text[])
            ORDER BY ts_last_active ASC
            """,
            ids,
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def delete_notes(ids: list[str]) -> int:
    """Delete agent notes matching the supplied identifiers."""
    conn = await asyncpg.connect(DB_URL)
    try:
        status = await conn.execute(
            f"""
            DELETE FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note' AND id = ANY($1::text[])
            """,
            ids,
        )
        return int(status.rsplit(" ", 1)[-1])
    finally:
        await conn.close()


async def find_duplicates(threshold: float, kind: str | None, limit: int) -> list[dict]:
    """Return active row pairs meeting the cosine-similarity threshold."""
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch(
            f"""
            SELECT a.id AS a_id, a.chunk_kind AS a_kind,
                   left(a.content_raw, {TEXT_LIMIT}) AS a_text,
                   b.id AS b_id, b.chunk_kind AS b_kind,
                   left(b.content_raw, {TEXT_LIMIT}) AS b_text,
                   1 - (a.embedding <=> b.embedding) AS score
            FROM "{PG_SCHEMA}".memory_chunks AS a
            JOIN "{PG_SCHEMA}".memory_chunks AS b ON a.id < b.id
            WHERE a.archived_at IS NULL
              AND b.archived_at IS NULL
              AND 1 - (a.embedding <=> b.embedding) >= $1
              AND ($2::text IS NULL OR (a.chunk_kind = $2 AND b.chunk_kind = $2))
            ORDER BY score DESC
            LIMIT $3
            """,
            threshold,
            kind,
            limit,
        )
        return [
            {
                "a": {"id": row["a_id"], "kind": row["a_kind"], "text": row["a_text"]},
                "b": {"id": row["b_id"], "kind": row["b_kind"], "text": row["b_text"]},
                "score": row["score"],
            }
            for row in rows
        ]
    finally:
        await conn.close()


async def archive_candidates(now: float) -> list[dict]:
    """Return active rows matching the configured cold-tier rule."""
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch(
            f"""
            SELECT id, chunk_kind AS kind,
                   ($1 - ts_last_active) / {DAY_SECONDS} AS age_days,
                   hit_count, last_hit_at
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE archived_at IS NULL
              AND ts_last_active < $1 - $2::double precision * {DAY_SECONDS}
              AND coalesce(last_hit_at, ts_last_active)
                  < $1 - $3::double precision * {DAY_SECONDS}
            ORDER BY ts_last_active ASC
            """,
            now,
            COLD_AGE_DAYS,
            COLD_UNHIT_DAYS,
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def archive_rows(ids: list[str], now: float) -> int:
    """Archive active rows matching the supplied identifiers."""
    conn = await asyncpg.connect(DB_URL)
    try:
        status = await conn.execute(
            f"""
            UPDATE "{PG_SCHEMA}".memory_chunks
            SET archived_at = $2
            WHERE id = ANY($1::text[]) AND archived_at IS NULL
            """,
            ids,
            now,
        )
        return int(status.rsplit(" ", 1)[-1])
    finally:
        await conn.close()


async def restore_rows(ids: list[str]) -> int:
    """Restore rows matching the supplied identifiers."""
    conn = await asyncpg.connect(DB_URL)
    try:
        status = await conn.execute(
            f"""
            UPDATE "{PG_SCHEMA}".memory_chunks
            SET archived_at = NULL
            WHERE id = ANY($1::text[])
            """,
            ids,
        )
        return int(status.rsplit(" ", 1)[-1])
    finally:
        await conn.close()


async def rows_by_ids(ids: list[str]) -> list[dict]:
    """Return lifecycle fields for rows matching the supplied identifiers."""
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch(
            f"""
            SELECT id, chunk_kind AS kind, archived_at, hit_count, last_hit_at
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE id = ANY($1::text[])
            ORDER BY id
            """,
            ids,
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()
