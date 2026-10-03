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
NOTE_KINDS = ("personal", "work")
NOTE_SIMILAR_THRESHOLD = float(os.getenv("NOTE_SIMILAR_THRESHOLD", "0.85"))
LIST_NOTES_DEFAULT_LIMIT = 50
LIST_NOTES_MAX_LIMIT = 200


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


_REFUSAL_RECOVERY = (
    "Rewrite it once to state what is worth keeping, or use send_message if it is progress "
    "or state for the next session. If the rewrite is refused too, store nothing and tell "
    "the user when one is present."
)


class LowSignalNoteError(ValueError):
    """The content gate judged a note's content not worth storing."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Refused by the memory gate: {reason} {_REFUSAL_RECOVERY}")


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
You judge notes for the long-term memory of one user, kept by the assistants and agents
that work with them: memory of the user and their life, and knowledge of their work that
a later session could not recover from the code, version control, the tracker,
documentation, or the running system. Accept a note that a later conversation or session
could use:

- who the user is, the people, places, and things around them, what they have and do,
  what they like or prefer and why, and how they want to be talked to;
- something that happened to the user or that they did, with its date and outcome, and
  their plans, goals, commitments, and choices;
- a change to something remembered earlier, with the new value and the old one;
- something the assistant made or gave the user that they may want again — a
  recommendation, a number, a list, a schedule, an arrangement — with its specifics;
- a decision about their work with its reason, a reproduced bug with its cause or fix, a
  non-obvious fact about how a machine, service, account, or tool behaves here, an
  approach that failed and why, or a convention the user set for how work is done;
- a moment in the relationship between the user and the assistant.

Refuse a note that is low signal:

- progress, status, next steps, or a narration of what was done in a session;
- a copy of what a record held elsewhere says — version control, the tracker, a file,
  documentation, the running system — its contents, scope, changes, or status. Something
  built in the conversation from facts no record holds has no source but the note and is
  not a copy;
- what a file or function does, or how a codebase is structured;
- generic advice or explanation with no fact tied to this user or their work;
- greetings, filler, or a restated question.

Whether a note is about the user's life or their work never refuses it, and neither does
a note that mixes the two. A passing event, a mood, or a one-off plan of the user is
memory. A failure's lesson is not session narration. The refusal list wins over the
accept list. State the reason in one sentence."""


async def judge_note_content(content: str) -> ContentVerdict:
    """Ask the chat model whether a note's content is worth keeping across sessions."""
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": content},
    ]
    verdict = await chat_json(messages, _VERDICT_SCHEMA, timeout=NOTE_GATE_TIMEOUT_SECONDS)
    return ContentVerdict(accepted=verdict["accepted"], reason=verdict["reason"])


async def _content_gate(row: dict[str, Any]) -> None:
    """Refuse a low-signal note before it costs an embedding call; fail open."""
    try:
        verdict = await judge_note_content(row["raw"])
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
    note_id = f"note:{namespace}:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
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
    await _content_gate(row)
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
                f"""
                INSERT INTO "{PG_SCHEMA}".memory_chunks
                  (id, source_type, source_ref, chunk_kind, session_id, content_raw,
                   distilled, embedding, ts_last_active, namespace, metadata, occurred_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::halfvec,$9,$10,$11::jsonb,$12)
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
                row["occurred_at"],
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
        if metadata.get("archived_by") is not None:
            note["archived_by"] = metadata["archived_by"]
        if metadata.get("supersedes") is not None:
            note["supersedes"] = metadata["supersedes"]
        out.append(note)
    return out
