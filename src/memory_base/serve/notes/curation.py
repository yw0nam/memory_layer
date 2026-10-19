"""Database operations for memory lifecycle management."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.retrieval.search import history_predicates, metadata_dict
from memory_base.serve.access import namespaces
from memory_base.serve.common.entities import (
    lock_slugs,
    require_registered,
    set_note_entities,
)
from memory_base.serve.common.http import TEXT_LIMIT
from memory_base.serve.notes.store import ARCHIVE_LINEAGE_FIELDS, MAX_NOTE_ENTITIES

COLD_AGE_DAYS = int(os.getenv("COLD_AGE_DAYS", "180"))
COLD_UNHIT_DAYS = int(os.getenv("COLD_UNHIT_DAYS", "90"))
DAY_SECONDS = 86400.0
DUPLICATE_NEIGHBORS = 5


def _scoped_row(row: Any) -> dict:
    """A lifecycle row carrying its author, and `shared` only when the flag is set."""
    out = dict(row)
    if not out.pop("shared", False):
        return out
    out["shared"] = True
    return out


def _shift_placeholders(clause: str, offset: int) -> str:
    """Move every `$n` placeholder in `clause` by `offset`."""
    return re.sub(r"\$(\d+)", lambda match: f"${int(match.group(1)) + offset}", clause)


async def list_old_notes(
    older_than_days: int, *, author: str | None = None, visible_to: str | None = None
) -> list[dict]:
    """Return active agent notes older than the requested age, within the read scope."""
    predicates, filter_args = history_predicates(
        include_archived=False, kind=None, tags=None, author=author, visible_to=visible_to
    )
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, content_raw AS text, ts_last_active, hit_count, last_hit_at,
                   metadata->>'author' AS author,
                   coalesce((metadata->>'shared')::boolean, false) AS shared
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note' AND {predicates}
              AND ts_last_active
                  < extract(epoch FROM now()) - $1::double precision * {DAY_SECONDS}
            ORDER BY ts_last_active ASC
            """,
            older_than_days,
            *filter_args,
        )
        return [_scoped_row(row) for row in rows]


async def notes_by_ids(ids: list[str]) -> list[dict]:
    """Return agent notes matching the supplied identifiers."""
    async with db.acquire() as conn:
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


async def delete_notes(ids: list[str]) -> int:
    """Delete agent notes matching the supplied identifiers.

    Current facts citing a deleted note are archived and lose their citation.
    """
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await lock_slugs(conn)
            await conn.execute(
                f"""
                UPDATE "{PG_SCHEMA}".facts SET archived_at = $2, note_id = NULL
                WHERE archived_at IS NULL AND note_id IN (
                  SELECT id FROM "{PG_SCHEMA}".memory_chunks
                  WHERE source_type = 'agent_note' AND id = ANY($1::text[])
                )
                """,
                ids,
                time.time(),
            )
            status = await conn.execute(
                f"""
                DELETE FROM "{PG_SCHEMA}".memory_chunks
                WHERE source_type = 'agent_note' AND id = ANY($1::text[])
                """,
                ids,
            )
            return int(status.rsplit(" ", 1)[-1])


async def move_notes(ids: list[str], target_namespace: str) -> dict[str, list]:
    """Move agent-note rows to another namespace, rewriting each id to its target format.

    An id already present in the target namespace is left alone and reported skipped
    instead of overwriting it; an id that names no agent_note row is skipped the same way.
    """
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await lock_slugs(conn)
            await namespaces.require_registered(conn, target_namespace)
            existing = await conn.fetch(
                f"""
                SELECT id FROM "{PG_SCHEMA}".memory_chunks
                WHERE source_type = 'agent_note' AND id = ANY($1::text[])
                """,
                ids,
            )
            found = {row["id"] for row in existing}
            moved: list[dict[str, str]] = []
            skipped: list[str] = []
            for old_id in ids:
                if old_id not in found:
                    skipped.append(old_id)
                    continue
                new_id = f"note:{target_namespace}:{old_id.rsplit(':', 1)[-1]}"
                collides = await conn.fetchval(
                    f'SELECT EXISTS(SELECT 1 FROM "{PG_SCHEMA}".memory_chunks WHERE id = $1)',
                    new_id,
                )
                if collides:
                    skipped.append(old_id)
                    continue
                await conn.execute(
                    f"""
                    UPDATE "{PG_SCHEMA}".memory_chunks
                    SET id = $1, session_id = $1, namespace = $2
                    WHERE id = $3
                    """,
                    new_id,
                    target_namespace,
                    old_id,
                )
                moved.append({"old": old_id, "new": new_id})
            return {"moved": moved, "skipped": skipped}


async def find_duplicates(
    threshold: float,
    kind: str | None,
    limit: int,
    *,
    author: str | None = None,
    visible_to: str | None = None,
) -> list[dict]:
    """Return active agent-note pairs meeting the cosine-similarity threshold.

    Both sides of a pair are inside the caller's scope.
    """
    outer, outer_args = history_predicates(
        include_archived=False,
        kind=None,
        tags=None,
        alias="a",
        author=author,
        visible_to=visible_to,
    )
    inner, inner_args = history_predicates(
        include_archived=False,
        kind=None,
        tags=None,
        alias="candidate",
        author=author,
        visible_to=visible_to,
    )
    # $1 threshold, $2 kind, and $3 limit come first; each clause follows them.
    outer_sql = _shift_placeholders(outer, 2)
    inner_sql = _shift_placeholders(inner, 2 + len(outer_args))
    args = [threshold, kind, limit, *outer_args, *inner_args]
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
                     CASE WHEN a.id < b.id THEN a.metadata->>'author'
                          ELSE b.metadata->>'author' END AS least_author,
                     CASE WHEN a.id < b.id THEN b.chunk_kind ELSE a.chunk_kind END
                       AS greatest_kind,
                     CASE WHEN a.id < b.id THEN left(b.content_raw, {TEXT_LIMIT})
                          ELSE left(a.content_raw, {TEXT_LIMIT}) END AS greatest_text,
                     CASE WHEN a.id < b.id THEN b.metadata->>'author'
                          ELSE a.metadata->>'author' END AS greatest_author,
                     1 - b.distance AS score
              FROM "{PG_SCHEMA}".memory_chunks AS a
              CROSS JOIN LATERAL (
                SELECT candidate.id, candidate.chunk_kind, candidate.content_raw,
                       candidate.metadata,
                       candidate.embedding <=> a.embedding AS distance
                FROM "{PG_SCHEMA}".memory_chunks AS candidate
                WHERE candidate.source_type = 'agent_note'
                  AND {inner_sql}
                  AND candidate.id <> a.id
                  AND ($2::text IS NULL OR candidate.chunk_kind = $2)
                ORDER BY candidate.embedding <=> a.embedding
                LIMIT {DUPLICATE_NEIGHBORS}
              ) AS b
              WHERE a.source_type = 'agent_note'
                AND {outer_sql}
                AND ($2::text IS NULL OR a.chunk_kind = $2)
            ),
            deduplicated AS (
              SELECT DISTINCT ON (least_id, greatest_id)
                     least_id AS a_id, least_kind AS a_kind, least_text AS a_text,
                     least_author AS a_author,
                     greatest_id AS b_id, greatest_kind AS b_kind,
                     greatest_text AS b_text, greatest_author AS b_author, score
              FROM directed
              WHERE score >= $1
              ORDER BY least_id, greatest_id, score DESC
            )
            SELECT a_id, a_kind, a_text, a_author, b_id, b_kind, b_text, b_author, score
            FROM deduplicated
            ORDER BY score DESC
            LIMIT $3
            """,
            *args,
        )
        return [
            {
                "a": {
                    "id": row["a_id"],
                    "kind": row["a_kind"],
                    "text": row["a_text"],
                    "author": row["a_author"],
                },
                "b": {
                    "id": row["b_id"],
                    "kind": row["b_kind"],
                    "text": row["b_text"],
                    "author": row["b_author"],
                },
                "score": row["score"],
            }
            for row in rows
        ]


async def archive_candidates(
    now: float, *, author: str | None = None, visible_to: str | None = None
) -> list[dict]:
    """Return active agent notes matching the configured cold-tier rule, within scope."""
    predicates, filter_args = history_predicates(
        include_archived=False, kind=None, tags=None, author=author, visible_to=visible_to
    )
    # $1 now, $2 COLD_AGE_DAYS, and $3 COLD_UNHIT_DAYS come before the filter args.
    predicates = _shift_placeholders(predicates, 2)
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, chunk_kind AS kind,
                   ($1 - ts_last_active) / {DAY_SECONDS} AS age_days,
                   hit_count, last_hit_at,
                   metadata->>'author' AS author,
                   coalesce((metadata->>'shared')::boolean, false) AS shared
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note' AND {predicates}
              AND ts_last_active < $1 - $2::double precision * {DAY_SECONDS}
              AND coalesce(last_hit_at, ts_last_active)
                  < $1 - $3::double precision * {DAY_SECONDS}
            ORDER BY ts_last_active ASC
            """,
            now,
            COLD_AGE_DAYS,
            COLD_UNHIT_DAYS,
            *filter_args,
        )
        return [_scoped_row(row) for row in rows]


async def archive_rows(
    ids: list[str],
    now: float,
    archived_by: str | None = None,
) -> int:
    """Archive active agent notes matching the supplied identifiers."""
    stamp = (
        ", metadata = metadata || jsonb_build_object('archived_by', $3::text)"
        if archived_by is not None
        else ""
    )
    extra = [archived_by] if archived_by is not None else []
    async with db.acquire() as conn, conn.transaction():
        await lock_slugs(conn)
        status = await conn.execute(
            f"""
            UPDATE "{PG_SCHEMA}".memory_chunks
            SET archived_at = $2{stamp}
            WHERE source_type = 'agent_note' AND id = ANY($1::text[]) AND archived_at IS NULL
            """,
            ids,
            now,
            *extra,
        )
        return int(status.rsplit(" ", 1)[-1])


async def restore_rows(ids: list[str]) -> int:
    """Restore rows matching the supplied identifiers.

    Restoring clears the fields that recorded the archive: its author and the note that
    replaced or absorbed the row.
    """
    cleared = "".join(f" - '{field}'" for field in ARCHIVE_LINEAGE_FIELDS)
    async with db.acquire() as conn, conn.transaction():
        await lock_slugs(conn)
        status = await conn.execute(
            f"""
            UPDATE "{PG_SCHEMA}".memory_chunks
            SET archived_at = NULL, metadata = metadata{cleared}
            WHERE id = ANY($1::text[])
            """,
            ids,
        )
        return int(status.rsplit(" ", 1)[-1])


async def rows_by_ids(ids: list[str], notes_only: bool = False) -> list[dict]:
    """Return lifecycle fields for rows matching the supplied identifiers.

    ``notes_only`` drops every row that is not an agent note.
    """
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, chunk_kind AS kind, archived_at, hit_count, last_hit_at,
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


class LinkRefused(ValueError):
    """A link_entities call refused as a whole; `ids` names the notes at fault."""

    def __init__(self, message: str, ids: list[str]) -> None:
        super().__init__(f"{message}: {', '.join(ids)}")
        self.ids = ids


async def link_entities(
    ids: list[str],
    add: list[str],
    remove: list[str],
    author: str,
    confirm: bool,
) -> dict[str, list[dict]]:
    """Apply (current + add) - remove to the entity slugs of active agent notes.

    Rows are locked in id order and the sets recomputed from them. A changed note records
    who set its entities and when; an unchanged note is not touched. Without `confirm`
    nothing is written.
    """
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await lock_slugs(conn)
            rows = await conn.fetch(
                f"""
                SELECT id, metadata FROM "{PG_SCHEMA}".memory_chunks
                WHERE source_type = 'agent_note' AND archived_at IS NULL
                  AND id = ANY($1::text[])
                ORDER BY id
                FOR UPDATE
                """,
                ids,
            )
            found = {row["id"]: metadata_dict(row["metadata"]) for row in rows}
            if missing := [i for i in ids if i not in found]:
                raise LinkRefused("not active notes", missing)
            await require_registered(conn, add)
            results = []
            for note_id in sorted(found):
                before = found[note_id].get("entities", [])
                after = [slug for slug in dict.fromkeys([*before, *add]) if slug not in remove]
                results.append(
                    {
                        "id": note_id,
                        "before": before,
                        "after": after,
                        "changed": set(before) != set(after),
                    }
                )
            if over := [r["id"] for r in results if len(r["after"]) > MAX_NOTE_ENTITIES]:
                raise LinkRefused(f"a note holds at most {MAX_NOTE_ENTITIES} entities", over)
            if confirm:
                now = time.time()
                for result in results:
                    if not result["changed"]:
                        continue
                    await set_note_entities(
                        conn, result["id"], found[result["id"]], result["after"], author, now
                    )
            return {"notes": results}
