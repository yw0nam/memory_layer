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
from memory_base.core.secrets import find_secret
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


def note_date(occurred_at: float | None, ts_last_active: float) -> str:
    """The day the remembered event happened when recorded, else the day it was saved."""
    ts = ts_last_active if occurred_at is None else occurred_at
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def link_fields(
    conversation_id: str | None, turn_start: int | None, turn_end: int | None
) -> dict[str, Any]:
    """A note's link to its conversation source, with only the parts that are set."""
    fields: dict[str, Any] = {}
    if conversation_id is not None:
        fields["conversation_id"] = conversation_id
    if turn_start is not None:
        fields["turn_start"] = turn_start
        fields["turn_end"] = turn_end
    return fields


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
            f"Refused by the content gate: {reason} If part of this note records something "
            "that exists nowhere else (a decision and what it ruled out, a stated constraint "
            "or preference, an observed environment fact, a lesson from a failure), rewrite it "
            "to state that fact directly, without restating its source (PR, issue, commit, "
            "file, tracker), and save that as a note of its own. If nothing in it does, store "
            "nothing; that is the expected outcome. Retry at most once: if the rewrite is "
            "refused too, do not save it, and tell the user when one is present."
        )


class CredentialNoteError(ValueError):
    """A note or one of its tags carries a credential."""

    def __init__(self, secret_type: str) -> None:
        self.secret_type = secret_type
        super().__init__(
            f"note contains a credential ({secret_type}); store the fact without the secret"
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
You judge notes for a long-term memory of one user's conversations, coding sessions and
personal chat alike. Accept a note when a future conversation would otherwise have to ask
again and it records:

- a durable fact about the user or the people, places, and things around them;
- what the user has, uses, does regularly, likes, dislikes, or plans, with dates when stated;
- a dated event the user took part in and its outcome;
- a decision and the reason for it, or the alternatives it ruled out;
- a specific answer the assistant gave that the user may ask for again — a recommendation, a
  number, a list, a schedule, the defining facts of something written for the user;
- a constraint, preference, environment fact, or lesson from a failure stated by a person;
- how the user's systems behave in use — limits, schedules, failure modes, fixes — stated by
  no record.

Refuse a note reporting:

- what a record held elsewhere says — version control, the tracker, the filesystem, the
  running system — its contents, scope, changes, or status; the record is the source, the
  note a copy;
- progress, status, or a narration of what was done in a coding session;
- what a file or function does;
- generic advice or explanation true for anyone, with no fact tied to this user or this
  conversation;
- greetings, filler, or a restated question.

A copy that carries what its source does not state still fails; that part goes in its own
note. A failure's lesson is not session narration; the failure and its fix, stated outright,
pass. An episode is judged only on provenance: lived by a person rather than recorded; its
moment never refuses it. State the reason in one sentence."""


async def judge_note_content(content: str, kind: str) -> ContentVerdict:
    """Ask the chat model whether a note's content is worth keeping across sessions."""
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"kind: {kind}\n\n{content}"},
    ]
    verdict = await chat_json(messages, _VERDICT_SCHEMA, timeout=NOTE_GATE_TIMEOUT_SECONDS)
    return ContentVerdict(accepted=verdict["accepted"], reason=verdict["reason"])


async def _content_gate(row: dict[str, Any]) -> None:
    """Refuse a low-signal note before it costs an embedding call; fail open."""
    try:
        verdict = await judge_note_content(row["raw"], row["kind"])
    except Exception as exc:
        logger.warning("content gate unavailable, saving anyway: {}", exc)
        row["metadata"]["content_gate"] = "unavailable"
        return
    if not verdict.accepted:
        raise LowSignalNoteError(verdict.reason)


def _turn_range(
    conversation_id: Any, turn_start: Any, turn_end: Any
) -> tuple[str | None, int | None, int | None]:
    if conversation_id is not None and (
        not isinstance(conversation_id, str) or not conversation_id.strip()
    ):
        raise ValueError("conversation_id must be a non-empty string")
    if turn_start is None and turn_end is None:
        return conversation_id, None, None
    if conversation_id is None:
        raise ValueError("turn_start/turn_end require a conversation_id")
    if turn_start is None or turn_end is None:
        raise ValueError("turn_start and turn_end must be given together")
    for bound in (turn_start, turn_end):
        if isinstance(bound, bool) or not isinstance(bound, int):
            raise ValueError("turn_start and turn_end must be integers")
    if not 0 <= turn_start <= turn_end:
        raise ValueError("turn_start must satisfy 0 <= turn_start <= turn_end")
    return conversation_id, turn_start, turn_end


def build_note_row(
    content: str,
    kind: str,
    tags: list[str],
    now: float,
    namespace: str = DEFAULT_NAMESPACE,
    author: str | None = None,
    *,
    conversation_id: str | None = None,
    turn_start: int | None = None,
    turn_end: int | None = None,
    occurred_at: float | None = None,
) -> dict[str, Any]:
    """Validate a note and map it to memory_chunks columns (no embedding).

    The id hashes the conversation id with the content, so identical text from
    two conversations stays two notes, and is qualified by the namespace.
    """
    content = content.strip()
    if not content:
        raise ValueError("content must not be empty")
    if len(content) > NOTE_MAX_CHARS:
        raise ValueError(f"content exceeds {NOTE_MAX_CHARS} chars")
    if kind not in NOTE_KINDS:
        raise ValueError(f"kind must be one of {NOTE_KINDS}")
    conversation_id, turn_start, turn_end = _turn_range(conversation_id, turn_start, turn_end)
    normalized_tags = normalize_tags([] if tags is None else tags)
    metadata: dict[str, Any] = {"tags": normalized_tags}
    if author is not None:
        metadata["author"] = author
    identity = f"{conversation_id or ''}\n{content}"
    note_id = f"note:{namespace}:{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
    return {
        "id": note_id,
        "source_type": "agent_note",
        "source_ref": "save_memory",
        "kind": kind,
        "session_id": note_id,
        "raw": content,
        "distilled": content,
        "timestamp": now,
        "metadata": metadata,
        "conversation_id": conversation_id,
        "turn_start": turn_start,
        "turn_end": turn_end,
        "occurred_at": occurred_at,
    }


async def _require_source(conn: Any, row: dict[str, Any], namespace: str) -> None:
    """The linked source must exist in the note's namespace and hold the turn range."""
    source = await conn.fetchrow(
        f"""
        SELECT namespace, jsonb_array_length(turns) AS turn_count
        FROM "{PG_SCHEMA}".conversation_sources
        WHERE id = $1
        FOR SHARE
        """,
        row["conversation_id"],
    )
    if source is None or source["namespace"] != namespace:
        raise ValueError(
            f"unknown conversation_id {row['conversation_id']!r} in namespace {namespace!r}; "
            "store the conversation source first"
        )
    if row["turn_end"] is not None and row["turn_end"] >= source["turn_count"]:
        raise ValueError(
            f"turn range {row['turn_start']}-{row['turn_end']} is outside conversation source "
            f"{row['conversation_id']}'s {source['turn_count']} turns"
        )


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
    conversation_id: str | None = None,
    turn_start: int | None = None,
    turn_end: int | None = None,
) -> dict[str, Any]:
    """Validate, embed, and idempotently store an agent-authored memory.

    `occurred_at` is an ISO 8601 date/datetime stored beside the save time, which
    stays the note's recency timestamp; a future or unparseable value raises
    ValueError. `conversation_id` links the note to a stored conversation source
    in the same namespace, optionally to its turns `turn_start`..`turn_end`.
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
        conversation_id=conversation_id,
        turn_start=turn_start,
        turn_end=turn_end,
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
    await _content_gate(row)
    embedding = await embed_text(VllmEmbedder(), row["raw"])
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await namespaces.require_registered(conn, namespace)
            if row["conversation_id"] is not None:
                await _require_source(conn, row, namespace)
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
                f"""
                INSERT INTO "{PG_SCHEMA}".memory_chunks
                  (id, source_type, source_ref, chunk_kind, session_id, content_raw,
                   distilled, embedding, ts_last_active, namespace, metadata,
                   conversation_id, source_turn_start, source_turn_end, occurred_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::halfvec,$9,$10,$11::jsonb,$12,$13,$14,$15)
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
                namespace,
                json.dumps(row["metadata"], ensure_ascii=False),
                row["conversation_id"],
                row["turn_start"],
                row["turn_end"],
                row["occurred_at"],
            )
            stored = status.endswith(" 1")
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
                   ts_last_active, namespace, archived_at, conversation_id,
                   source_turn_start, source_turn_end, occurred_at
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
        note.update(
            link_fields(row["conversation_id"], row["source_turn_start"], row["source_turn_end"])
        )
        if row["archived_at"] is not None:
            note["archived"] = True
        if metadata.get("archived_by") is not None:
            note["archived_by"] = metadata["archived_by"]
        if metadata.get("supersedes") is not None:
            note["supersedes"] = metadata["supersedes"]
        out.append(note)
    return out
