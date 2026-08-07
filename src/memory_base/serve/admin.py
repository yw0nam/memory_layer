"""Database operations for memory lifecycle management."""

from __future__ import annotations

import os

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.serve.http import TEXT_LIMIT

COLD_AGE_DAYS = int(os.getenv("COLD_AGE_DAYS", "180"))
COLD_UNHIT_DAYS = int(os.getenv("COLD_UNHIT_DAYS", "90"))
DAY_SECONDS = 86400.0
DUPLICATE_NEIGHBORS = 5


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


async def list_old_notes(older_than_days: int, namespaces: list[str] | None = None) -> list[dict]:
    """Return active agent notes older than the requested age, scoped to namespaces (None: all)."""
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, content_raw AS text, ts_last_active, hit_count, last_hit_at
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note'
              AND archived_at IS NULL
              AND ts_last_active
                  < extract(epoch FROM now()) - $1::double precision * {DAY_SECONDS}
              AND ($2::text[] IS NULL OR namespace = ANY($2::text[]))
            ORDER BY ts_last_active ASC
            """,
            older_than_days,
            namespaces,
        )
        return [dict(row) for row in rows]


async def notes_by_ids(ids: list[str], namespaces: list[str] | None = None) -> list[dict]:
    """Return agent notes matching the supplied identifiers, scoped to namespaces (None: all)."""
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, content_raw AS text, ts_last_active, hit_count, last_hit_at,
                   archived_at
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note' AND id = ANY($1::text[])
              AND ($2::text[] IS NULL OR namespace = ANY($2::text[]))
            ORDER BY ts_last_active ASC
            """,
            ids,
            namespaces,
        )
        return [dict(row) for row in rows]


async def delete_notes(ids: list[str], namespaces: list[str] | None = None) -> int:
    """Delete agent notes matching the supplied identifiers, scoped to namespaces (None: all)."""
    async with db.acquire() as conn:
        status = await conn.execute(
            f"""
            DELETE FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note' AND id = ANY($1::text[])
              AND ($2::text[] IS NULL OR namespace = ANY($2::text[]))
            """,
            ids,
            namespaces,
        )
        return int(status.rsplit(" ", 1)[-1])


async def find_duplicates(
    threshold: float, kind: str | None, limit: int, namespaces: list[str] | None = None
) -> list[dict]:
    """Return active row pairs meeting the cosine-similarity threshold, scoped to namespaces."""
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            WITH directed AS (
              SELECT least(a.id, b.id) AS least_id,
                     greatest(a.id, b.id) AS greatest_id,
                     CASE WHEN a.id < b.id THEN a.chunk_kind ELSE b.chunk_kind END
                       AS least_kind,
                     CASE WHEN a.id < b.id THEN left(a.content_raw, {TEXT_LIMIT})
                          ELSE left(b.content_raw, {TEXT_LIMIT}) END AS least_text,
                     CASE WHEN a.id < b.id THEN b.chunk_kind ELSE a.chunk_kind END
                       AS greatest_kind,
                     CASE WHEN a.id < b.id THEN left(b.content_raw, {TEXT_LIMIT})
                          ELSE left(a.content_raw, {TEXT_LIMIT}) END AS greatest_text,
                     1 - b.distance AS score
              FROM "{PG_SCHEMA}".memory_chunks AS a
              CROSS JOIN LATERAL (
                SELECT candidate.id, candidate.chunk_kind, candidate.content_raw,
                       candidate.embedding <=> a.embedding AS distance
                FROM "{PG_SCHEMA}".memory_chunks AS candidate
                WHERE candidate.archived_at IS NULL
                  AND candidate.id <> a.id
                  AND ($2::text IS NULL OR candidate.chunk_kind = $2)
                  AND ($4::text[] IS NULL OR candidate.namespace = ANY($4::text[]))
                ORDER BY candidate.embedding <=> a.embedding
                LIMIT {DUPLICATE_NEIGHBORS}
              ) AS b
              WHERE a.archived_at IS NULL
                AND ($2::text IS NULL OR a.chunk_kind = $2)
                AND ($4::text[] IS NULL OR a.namespace = ANY($4::text[]))
            ),
            deduplicated AS (
              SELECT DISTINCT ON (least_id, greatest_id)
                     least_id AS a_id, least_kind AS a_kind, least_text AS a_text,
                     greatest_id AS b_id, greatest_kind AS b_kind,
                     greatest_text AS b_text, score
              FROM directed
              WHERE score >= $1
              ORDER BY least_id, greatest_id, score DESC
            )
            SELECT a_id, a_kind, a_text, b_id, b_kind, b_text, score
            FROM deduplicated
            ORDER BY score DESC
            LIMIT $3
            """,
            threshold,
            kind,
            limit,
            namespaces,
        )
        return [
            {
                "a": {"id": row["a_id"], "kind": row["a_kind"], "text": row["a_text"]},
                "b": {"id": row["b_id"], "kind": row["b_kind"], "text": row["b_text"]},
                "score": row["score"],
            }
            for row in rows
        ]


async def archive_candidates(now: float, namespaces: list[str] | None = None) -> list[dict]:
    """Return active rows matching the configured cold-tier rule, scoped to namespaces."""
    async with db.acquire() as conn:
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
              AND ($4::text[] IS NULL OR namespace = ANY($4::text[]))
            ORDER BY ts_last_active ASC
            """,
            now,
            COLD_AGE_DAYS,
            COLD_UNHIT_DAYS,
            namespaces,
        )
        return [dict(row) for row in rows]


async def archive_rows(ids: list[str], now: float, namespaces: list[str] | None = None) -> int:
    """Archive active rows matching the supplied identifiers, scoped to namespaces."""
    async with db.acquire() as conn:
        status = await conn.execute(
            f"""
            UPDATE "{PG_SCHEMA}".memory_chunks
            SET archived_at = $2
            WHERE id = ANY($1::text[]) AND archived_at IS NULL
              AND ($3::text[] IS NULL OR namespace = ANY($3::text[]))
            """,
            ids,
            now,
            namespaces,
        )
        return int(status.rsplit(" ", 1)[-1])


async def restore_rows(ids: list[str], namespaces: list[str] | None = None) -> int:
    """Restore rows matching the supplied identifiers, scoped to namespaces."""
    async with db.acquire() as conn:
        status = await conn.execute(
            f"""
            UPDATE "{PG_SCHEMA}".memory_chunks
            SET archived_at = NULL
            WHERE id = ANY($1::text[])
              AND ($2::text[] IS NULL OR namespace = ANY($2::text[]))
            """,
            ids,
            namespaces,
        )
        return int(status.rsplit(" ", 1)[-1])


async def rows_by_ids(ids: list[str], namespaces: list[str] | None = None) -> list[dict]:
    """Return lifecycle fields for rows matching the supplied identifiers, scoped to namespaces."""
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, chunk_kind AS kind, archived_at, hit_count, last_hit_at
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE id = ANY($1::text[])
              AND ($2::text[] IS NULL OR namespace = ANY($2::text[]))
            ORDER BY id
            """,
            ids,
            namespaces,
        )
        return [dict(row) for row in rows]
