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
from memory_base.core.secrets import find_secret
from memory_base.retrieval.search import (
    history_predicates,
    metadata_dict,
    normalize_tags,
    normalize_time_range,
    parse_time_bound,
)
from memory_base.serve import namespaces
from memory_base.serve.common.http import TEXT_LIMIT
from memory_base.serve.namespaces import DEFAULT_NAMESPACE


NOTE_MAX_CHARS = 4000
NOTE_KINDS = ("personal", "work")
NOTE_SIMILAR_THRESHOLD = float(os.getenv("NOTE_SIMILAR_THRESHOLD", "0.85"))
LIST_NOTES_DEFAULT_LIMIT = 50
LIST_NOTES_MAX_LIMIT = 200
# Written when a note is archived by a save, an archive, or a consolidation; a restore clears them.
ARCHIVE_LINEAGE_FIELDS = ("archived_by", "replaced_by", "consolidated_into")
LINEAGE_FIELDS = (
    "supersedes",
    *ARCHIVE_LINEAGE_FIELDS,
    "merged_from",
    "merged_dates",
    "consolidation_action",
    "undone_action",
)


def note_date(occurred_at: float | None, ts_last_active: float) -> str:
    """The day the remembered event happened when recorded, else the day it was saved."""
    ts = ts_last_active if occurred_at is None else occurred_at
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


class SimilarNotesError(ValueError):
    """A new note landed next to near-identical active notes without resolving them."""

    def __init__(self, similar: list[dict[str, Any]]) -> None:
        self.similar = similar
        listing = "\n".join(f"  {s['id']} ({s['score']:.2f}): {s['text'][:300]}" for s in similar)
        super().__init__(
            f"Refused: {len(similar)} active note(s) in this namespace say nearly the same thing. "
            "Read them. If this note replaces one, call again with supersedes=<id> so the old one "
            "is archived. Only if it records a genuinely different fact, call again with "
            f"allow_similar=true.\n{listing}"
        )


class CredentialNoteError(ValueError):
    """A note or one of its tags carries a credential."""

    def __init__(self, secret_type: str) -> None:
        self.secret_type = secret_type
        super().__init__(
            f"note contains a credential ({secret_type}); store the fact without the secret"
        )


def note_id(namespace: str, content: str) -> str:
    """`note:{namespace}:{sha256(content)[:16]}` over the stripped content."""
    return f"note:{namespace}:{hashlib.sha256(content.strip().encode()).hexdigest()[:16]}"


INSERT_NOTE_SQL = f"""
INSERT INTO "{PG_SCHEMA}".memory_chunks
  (id, source_type, source_ref, chunk_kind, session_id, content_raw,
   distilled, embedding, ts_last_active, namespace, metadata, occurred_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8::halfvec,$9,$10,$11::jsonb,$12)
ON CONFLICT (id) DO NOTHING
"""


def insert_note_args(row: dict[str, Any], embedding: str, namespace: str) -> tuple[Any, ...]:
    """The INSERT_NOTE_SQL arguments for a `build_note_row` row."""
    return (
        row["id"],
        row["source_type"],
        row["source_ref"],
        row["kind"],
        row["session_id"],
        row["raw"],
        row["distilled"],
        embedding,
        row["timestamp"],
        namespace,
        json.dumps(row["metadata"], ensure_ascii=False),
        row["occurred_at"],
    )


def build_note_row(
    content: str,
    kind: str,
    tags: list[str],
    now: float,
    namespace: str = DEFAULT_NAMESPACE,
    author: str | None = None,
    *,
    occurred_at: float | None = None,
) -> dict[str, Any]:
    """Validate a note and map it to memory_chunks columns (no embedding).

    The id is `note:{namespace}:{sha256(content)[:16]}` over the stripped content.
    """
    content = content.strip()
    if not content:
        raise ValueError("content must not be empty")
    if len(content) > NOTE_MAX_CHARS:
        raise ValueError(f"content exceeds {NOTE_MAX_CHARS} chars")
    if kind not in NOTE_KINDS:
        raise ValueError(f"kind must be one of {NOTE_KINDS}")
    normalized_tags = normalize_tags([] if tags is None else tags)
    metadata: dict[str, Any] = {"tags": normalized_tags}
    if author is not None:
        metadata["author"] = author
    row_id = note_id(namespace, content)
    return {
        "id": row_id,
        "source_type": "agent_note",
        "source_ref": "save_memory",
        "kind": kind,
        "session_id": row_id,
        "raw": content,
        "distilled": content,
        "timestamp": now,
        "metadata": metadata,
        "occurred_at": occurred_at,
    }


async def save_note(
    content: str,
    *,
    kind: str,
    tags: list[str],
    supersedes: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    occurred_at: str | None = None,
    author: str | None = None,
    allow_similar: bool = False,
) -> dict[str, Any]:
    """Validate, embed, and idempotently store an agent-authored memory.

    `occurred_at` is an ISO 8601 date/datetime stored beside the save time, which
    stays the note's recency timestamp; a future or unparseable value raises
    ValueError.
    """
    now = time.time()
    occurred_ts = parse_time_bound(occurred_at) if occurred_at is not None else None
    if occurred_ts is not None and occurred_ts > now:
        raise ValueError("occurred_at must not be in the future")
    row = build_note_row(
        content,
        kind,
        tags,
        now,
        namespace,
        author,
        occurred_at=occurred_ts,
    )
    secret_type = find_secret("\n".join([content, *(tags or [])]))
    if secret_type is not None:
        raise CredentialNoteError(secret_type)
    if supersedes == row["id"]:
        raise ValueError(
            f"content is identical to the note it supersedes ({supersedes}), so there is "
            "nothing to replace; change the content or drop supersedes"
        )
    embedding = await embed_text(VllmEmbedder(), row["raw"])
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await namespaces.require_registered(conn, namespace)
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

            neighbours = [
                dict(neighbour)
                for neighbour in await conn.fetch(
                    f"""
                    SELECT id, 1 - (embedding <=> $1::halfvec) AS score,
                           left(content_raw, {TEXT_LIMIT}) AS text
                    FROM "{PG_SCHEMA}".memory_chunks
                    WHERE source_type = 'agent_note'
                      AND archived_at IS NULL
                      AND namespace = $4
                      AND id <> $2
                      AND 1 - (embedding <=> $1::halfvec) > $3
                    ORDER BY score DESC
                    LIMIT 3
                    """,
                    embedding,
                    row["id"],
                    NOTE_SIMILAR_THRESHOLD,
                    namespace,
                )
            ]
            acknowledged = [n["id"] for n in neighbours if n["id"] != supersedes]
            if allow_similar and acknowledged:
                row["metadata"]["similar_ack"] = acknowledged
            if supersedes is not None:
                row["metadata"]["supersedes"] = supersedes
            status = await conn.execute(
                INSERT_NOTE_SQL, *insert_note_args(row, embedding, namespace)
            )
            stored = status.endswith(" 1")
            stored_kind = row["kind"]
            if not stored:
                stored_kind = await conn.fetchval(
                    f'SELECT chunk_kind FROM "{PG_SCHEMA}".memory_chunks WHERE id = $1',
                    row["id"],
                )
            if (
                stored
                and neighbours
                and not allow_similar
                and supersedes not in {n["id"] for n in neighbours}
            ):
                raise SimilarNotesError(neighbours)
            if not stored and supersedes is not None:
                active = await conn.fetchval(
                    f"""
                    SELECT EXISTS(
                      SELECT 1 FROM "{PG_SCHEMA}".memory_chunks
                      WHERE id = $1 AND archived_at IS NULL
                    )
                    """,
                    row["id"],
                )
                if not active:
                    raise ValueError(
                        f"content is identical to archived note {row['id']}; restore it with "
                        f"restore_notes instead of re-saving it, then archive {supersedes} "
                        "with archive_notes"
                    )
            if supersedes is not None:
                await conn.execute(
                    f"""
                    UPDATE "{PG_SCHEMA}".memory_chunks
                    SET archived_at = $2,
                        metadata = metadata
                          || jsonb_build_object('archived_by', $4::text, 'replaced_by', $5::text)
                    WHERE id = $1 AND namespace = $3
                    """,
                    supersedes,
                    row["timestamp"],
                    namespace,
                    author,
                    row["id"],
                )
    return {
        "id": row["id"],
        "kind": stored_kind,
        "stored": stored,
        "superseded": supersedes,
        "similar": [n for n in neighbours if n["id"] != supersedes],
    }


async def list_notes(
    *,
    tags: list[str] | None = None,
    kind: str | None = None,
    namespaces: list[str] | None = None,
    include_archived: bool = False,
    since: str | None = None,
    until: str | None = None,
    author: str | None = None,
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
        author=author,
    )
    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, chunk_kind AS kind, content_raw AS text, metadata,
                   ts_last_active, namespace, archived_at, occurred_at
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
        metadata = metadata_dict(row["metadata"])
        note = {
            "id": row["id"],
            "kind": row["kind"],
            "text": row["text"][:TEXT_LIMIT],
            "tags": metadata.get("tags", []),
            "author": metadata.get("author"),
            "namespace": row["namespace"],
            "date": note_date(row["occurred_at"], row["ts_last_active"]),
        }
        if row["archived_at"] is not None:
            note["archived"] = True
        for field in LINEAGE_FIELDS:
            if metadata.get(field) is not None:
                note[field] = metadata[field]
        out.append(note)
    return out
