"""Validation and storage for agent-authored memory notes."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, VllmEmbedder, embed_text
from memory_base.core.llm import chat_json
from memory_base.core.schema import ensure_schema_once
from memory_base.retrieval.search import (
    history_predicates,
    metadata_dict,
    normalize_tags,
    normalize_time_range,
    parse_time_bound,
)
from memory_base.serve import namespaces
from memory_base.serve.http import TEXT_LIMIT
from memory_base.serve.namespaces import DEFAULT_NAMESPACE


NOTE_MAX_CHARS = 4000
NOTE_GATE_TIMEOUT_SECONDS = 20.0
NOTE_KINDS = ("note", "decision", "episode")
NOTE_SIMILAR_THRESHOLD = float(os.getenv("NOTE_SIMILAR_THRESHOLD", "0.85"))
LIST_NOTES_DEFAULT_LIMIT = 50
LIST_NOTES_MAX_LIMIT = 200


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


class LowSignalNoteError(ValueError):
    """The content gate judged a note's content not worth storing."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            f"Refused: {reason} If it does record something a future session could not "
            "recover from anywhere else, call again with allow_restatement=true."
        )


@dataclass(frozen=True)
class ContentVerdict:
    """The chat model's judgement of one note's content."""

    accepted: bool
    reason: str


_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"accepted": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["accepted", "reason"],
}

JUDGE_PROMPT = """\
You decide whether a note belongs in a long-term memory store that later sessions read.
The test is provenance, not usefulness. Accept a note only when what it records exists
nowhere else — a conclusion and what it ruled out, a constraint or preference someone
stated, an observed fact about the environment, a lesson from something that failed — and
a future reader would need it in order to act well. Refuse a note that reports what a
record held elsewhere says — its contents, scope, changes, or status — however detailed
the report: that record is the source and the note is a copy. Version control, the
tracker, the filesystem, and the running system are such records. Refuse what is bound to
the moment it was written: progress, status, a narration of what was done. A copy that
also carries something its source does not state still fails; that part belongs in a note
of its own. Judge the content alone; where the note is stored, whom it concerns, and what
domain it comes from are not your concern. An episode is the one kind that records a
dated event, so being bound to its moment is never itself a reason to refuse one: judge an
episode on provenance alone, accepting an event someone lived through and refusing one a
tracker, version control, or the filesystem already records. State the deciding reason in
one sentence."""


async def judge_note_content(content: str, kind: str) -> ContentVerdict:
    """Ask the chat model whether a note's content is worth keeping across sessions."""
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"kind: {kind}\n\n{content}"},
    ]
    verdict = await chat_json(messages, _VERDICT_SCHEMA, timeout=NOTE_GATE_TIMEOUT_SECONDS)
    return ContentVerdict(accepted=verdict["accepted"], reason=verdict["reason"])


async def _content_gate(row: dict[str, Any], allow_restatement: bool) -> None:
    """Refuse a low-signal note before it costs an embedding call; fail open."""
    if allow_restatement:
        row["metadata"]["content_gate"] = "overridden"
        return
    try:
        verdict = await judge_note_content(row["raw"], row["kind"])
    except Exception as exc:
        logger.warning("content gate unavailable, saving anyway: {}", exc)
        row["metadata"]["content_gate"] = "unavailable"
        return
    if not verdict.accepted:
        raise LowSignalNoteError(verdict.reason)


def build_note_row(
    content: str,
    kind: str,
    tags: list[str],
    now: float,
    namespace: str = DEFAULT_NAMESPACE,
    author: str | None = None,
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
    normalized_tags = normalize_tags([] if tags is None else tags)
    metadata: dict[str, Any] = {"tags": normalized_tags}
    if author is not None:
        metadata["author"] = author
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
        "metadata": metadata,
    }


async def save_note(
    content: str,
    *,
    tags: list[str],
    kind: str = "note",
    supersedes: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    occurred_at: str | None = None,
    author: str | None = None,
    allow_similar: bool = False,
    allow_restatement: bool = False,
) -> dict[str, Any]:
    """Validate, embed, and idempotently store an agent-authored memory.

    `occurred_at`, when given, backdates the stored timestamp to that ISO 8601
    date/datetime instead of now; a future or unparseable value raises ValueError.
    """
    now = time.time()
    ts = parse_time_bound(occurred_at) if occurred_at is not None else now
    if ts > now:
        raise ValueError("occurred_at must not be in the future")
    row = build_note_row(content, kind, tags, ts, namespace, author)
    await _content_gate(row, allow_restatement)
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
            stored = status.endswith(" 1")
            if (
                stored
                and neighbours
                and not allow_similar
                and supersedes not in {n["id"] for n in neighbours}
            ):
                raise SimilarNotesError(neighbours)
            if supersedes is not None:
                await conn.execute(
                    f"""
                    UPDATE "{PG_SCHEMA}".memory_chunks
                    SET archived_at = $2,
                        metadata = metadata || jsonb_build_object('archived_by', $4::text)
                    WHERE id = $1 AND namespace = $3
                    """,
                    supersedes,
                    row["timestamp"],
                    namespace,
                    author,
                )
    return {
        "id": row["id"],
        "kind": row["kind"],
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
        metadata = metadata_dict(row["metadata"])
        note = {
            "id": row["id"],
            "kind": row["kind"],
            "text": row["text"][:TEXT_LIMIT],
            "tags": metadata.get("tags", []),
            "author": metadata.get("author"),
            "namespace": row["namespace"],
            "date": datetime.fromtimestamp(row["ts_last_active"], tz=timezone.utc).strftime(
                "%Y-%m-%d"
            ),
        }
        if row["archived_at"] is not None:
            note["archived"] = True
        if metadata.get("archived_by") is not None:
            note["archived_by"] = metadata["archived_by"]
        out.append(note)
    return out
