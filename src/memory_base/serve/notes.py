"""Validation and storage for agent-authored memory notes."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, VllmEmbedder, embed_text
from memory_base.core.schema import ensure_schema_once
from memory_base.retrieval.search import (
    history_predicates,
    metadata_dict,
    normalize_tags,
    normalize_time_range,
)
from memory_base.serve import namespaces
from memory_base.serve.http import TEXT_LIMIT
from memory_base.serve.namespaces import DEFAULT_NAMESPACE

NOTE_MAX_CHARS = 4000
NOTE_KINDS = ("note", "decision")
NOTE_SIMILAR_THRESHOLD = float(os.getenv("NOTE_SIMILAR_THRESHOLD", "0.85"))
LIST_NOTES_DEFAULT_LIMIT = 50
LIST_NOTES_MAX_LIMIT = 200


def build_note_row(
    content: str,
    kind: str,
    tags: list[str] | None,
    now: float,
    namespace: str = DEFAULT_NAMESPACE,
) -> dict[str, Any]:
    """Validate a note and map it to memory_chunks columns (no embedding).

    The id is namespace-qualified for every namespace but 'default', so
    identical content saved into different namespaces gets independent rows
    instead of colliding on id and silently no-opping the second save. The
    'default' format stays exactly as before, preserving legacy idempotency.
    """
    content = content.strip()
    if not content:
        raise ValueError("content must not be empty")
    if len(content) > NOTE_MAX_CHARS:
        raise ValueError(f"content exceeds {NOTE_MAX_CHARS} chars")
    if kind not in NOTE_KINDS:
        raise ValueError(f"kind must be one of {NOTE_KINDS}")
    normalized_tags = normalize_tags(tags, allow_empty=True)
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    if namespace == DEFAULT_NAMESPACE:
        note_id = f"note:{content_hash}"
    else:
        note_id = f"note:{namespace}:{content_hash}"
    return {
        "id": note_id,
        "source_type": "agent_note",
        "source_ref": "save_memory",
        "kind": kind,
        "session_id": note_id,
        "raw": content,
        "distilled": content,
        "timestamp": now,
        "idf": None,
        "metadata": {"tags": normalized_tags} if normalized_tags else {},
    }


async def save_note(
    content: str,
    kind: str = "note",
    tags: list[str] | None = None,
    supersedes: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> dict[str, Any]:
    """Validate, embed, and idempotently store an agent-authored memory."""
    row = build_note_row(content, kind, tags, time.time(), namespace)
    embedding = await embed_text(VllmEmbedder(), row["raw"])
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        await namespaces.require_registered(conn, namespace)
        async with conn.transaction():
            if supersedes is not None:
                exists = await conn.fetchval(
                    f"""
                    SELECT EXISTS(
                      SELECT 1 FROM "{PG_SCHEMA}".memory_chunks
                      WHERE id = $1 AND source_type = 'agent_note' AND namespace = $2
                    )
                    """,
                    supersedes,
                    namespace,
                )
                if not exists:
                    raise ValueError(f"unknown supersedes id: {supersedes}")

            status = await conn.execute(
                f"""
                INSERT INTO "{PG_SCHEMA}".memory_chunks
                  (id, source_type, source_ref, chunk_kind, session_id, content_raw,
                   distilled, embedding, ts_last_active, idf_score, namespace, metadata)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::halfvec,$9,$10,$11,$12::jsonb)
                ON CONFLICT (id) DO NOTHING
                """,
                row["id"],
                row["source_type"],
                row["source_ref"],
                row["kind"],
                row["session_id"],
                row["raw"],
                row["distilled"],
                embedding,
                row["timestamp"],
                row["idf"],
                namespace,
                json.dumps(row["metadata"], ensure_ascii=False),
            )
            if supersedes is not None:
                await conn.execute(
                    f"""
                    UPDATE "{PG_SCHEMA}".memory_chunks
                    SET archived_at = $2
                    WHERE id = $1 AND namespace = $3
                    """,
                    supersedes,
                    row["timestamp"],
                    namespace,
                )
            excluded_ids = [row["id"]]
            if supersedes is not None:
                excluded_ids.append(supersedes)
            similar_rows = await conn.fetch(
                f"""
                SELECT id, 1 - (embedding <=> $1::halfvec) AS score,
                       left(content_raw, {TEXT_LIMIT}) AS text
                FROM "{PG_SCHEMA}".memory_chunks
                WHERE source_type = 'agent_note'
                  AND archived_at IS NULL
                  AND namespace = $4
                  AND id <> ALL($2::text[])
                  AND 1 - (embedding <=> $1::halfvec) > $3
                ORDER BY score DESC
                LIMIT 3
                """,
                embedding,
                excluded_ids,
                NOTE_SIMILAR_THRESHOLD,
                namespace,
            )
    return {
        "id": row["id"],
        "kind": row["kind"],
        "stored": status.endswith(" 1"),
        "superseded": supersedes,
        "similar": [dict(similar) for similar in similar_rows],
    }


async def list_notes(
    *,
    tags: list[str] | None = None,
    kind: str | None = None,
    namespaces: list[str] | None = None,
    include_archived: bool = False,
    since: str | None = None,
    until: str | None = None,
    limit: int = LIST_NOTES_DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """List agent notes newest-first without a search query or embedding call.

    Every filter is optional; `namespaces` of None means every namespace (the
    caller resolves permission scope before calling).
    """
    if kind is not None and kind not in NOTE_KINDS:
        raise ValueError(f"kind must be one of {NOTE_KINDS}")
    normalized_tags = normalize_tags(tags)
    since_ts, until_ts = normalize_time_range(since, until)
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= LIST_NOTES_MAX_LIMIT
    ):
        raise ValueError(f"limit must be an integer between 1 and {LIST_NOTES_MAX_LIMIT}")
    predicates, filter_args = history_predicates(
        include_archived=include_archived,
        kind=kind,
        tags=normalized_tags,
        namespaces=namespaces,
        since=since_ts,
        until=until_ts,
    )
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, chunk_kind AS kind, content_raw AS text, metadata,
                   ts_last_active, namespace, archived_at
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'agent_note' AND {predicates}
            ORDER BY ts_last_active DESC
            LIMIT $1
            """,
            limit,
            *filter_args,
        )
    out: list[dict[str, Any]] = []
    for row in rows:
        note = {
            "id": row["id"],
            "kind": row["kind"],
            "text": row["text"][:TEXT_LIMIT],
            "tags": metadata_dict(row["metadata"]).get("tags", []),
            "namespace": row["namespace"],
            "date": datetime.fromtimestamp(row["ts_last_active"], tz=timezone.utc).strftime(
                "%Y-%m-%d"
            ),
        }
        if row["archived_at"] is not None:
            note["archived"] = True
        out.append(note)
    return out
