"""Conversation sources: the user and assistant turns of one session, stored unembedded.

A source is evidence a note points back to, read by address and turn range only:
it is never embedded, never indexed for BM25, and never read by search. The id is
derived from (namespace, origin, external_session_id), so re-uploading a session
replaces its turns in place until a note references the source or a distill job has
consumed some of them; from then on the stored turns are fixed and a re-upload may
only append turns after them. Every upload admits a distill job inside the upload's
transaction.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.core.secrets import find_secret
from memory_base.serve import job_store, namespaces
from memory_base.serve.distill import ORIGIN_PROMPTS

CONVERSATION_MAX_CHARS = 2_000_000
IDENTIFIER_MAX_CHARS = 256
METADATA_MAX_BYTES = 2048
TURN_ROLES = ("user", "assistant")
TURN_KEYS = frozenset({"role", "text"})


class ConversationTooLarge(ValueError):
    """The turns' total text exceeds CONVERSATION_MAX_CHARS."""


class ConversationConflict(Exception):
    """A re-upload would change the stored turns of a referenced or distilled source."""


class ConversationForbidden(Exception):
    """Only the key that first stored a source, or an admin, may replace it."""


class ConversationNotFound(Exception):
    """No source with that id is readable by the caller."""


def conversation_source_id(namespace: str, origin: str, external_session_id: str) -> str:
    digest = hashlib.sha256(f"{namespace}\n{origin}\n{external_session_id}".encode()).hexdigest()
    return f"conv:{digest[:16]}"


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(value) > IDENTIFIER_MAX_CHARS:
        raise ValueError(f"{field} must be at most {IDENTIFIER_MAX_CHARS} chars")
    return value


def _epoch(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be epoch seconds")
    return float(value)


def validate_turns(turns: Any) -> list[dict[str, str]]:
    """Each turn is exactly {role, text} and carries no credential; tool output never enters."""
    if not isinstance(turns, list) or not turns:
        raise ValueError("turns must be a non-empty list of {role, text} objects")
    total = 0
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict) or set(turn) != TURN_KEYS:
            raise ValueError(f"turn {index} must have exactly the keys role and text")
        if turn["role"] not in TURN_ROLES:
            raise ValueError(f"turn {index} role must be one of {TURN_ROLES}")
        text = turn["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"turn {index} text must be a non-empty string")
        total += len(text)
    if total > CONVERSATION_MAX_CHARS:
        raise ConversationTooLarge(
            f"turns total {total} chars, over the {CONVERSATION_MAX_CHARS} char limit"
        )
    for index, turn in enumerate(turns):
        secret_type = find_secret(turn["text"])
        if secret_type is not None:
            raise ValueError(
                f"turn {index} contains a credential ({secret_type}); remove it and upload again"
            )
    return [{"role": turn["role"], "text": turn["text"]} for turn in turns]


def validate_metadata(metadata: Any) -> str:
    """The client's own facts about the session as a JSON object of at most 2 KB."""
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    payload = json.dumps(metadata, ensure_ascii=False)
    if len(payload.encode()) > METADATA_MAX_BYTES:
        raise ValueError(f"metadata must be at most {METADATA_MAX_BYTES} bytes serialized")
    secret_type = find_secret(payload)
    if secret_type is not None:
        raise ValueError(f"metadata contains a credential ({secret_type}); remove it")
    return payload


async def _admit(conn: Any, key: Any, namespace: str, origin: str, source_id: str) -> str | None:
    if origin not in ORIGIN_PROMPTS:
        return None
    job = await job_store.admit_conversation(
        job_id=uuid.uuid4().hex,
        key_id=key.key_id,
        key_label=key.label,
        namespace=namespace,
        conversation_id=source_id,
        connection=conn,
    )
    return job.job_id


async def store_conversation(
    key: Any,
    *,
    namespace: str,
    origin: Any,
    external_session_id: Any,
    started_at: Any,
    ended_at: Any,
    turns: Any,
    metadata: Any = None,
) -> dict[str, Any]:
    """Insert or update a source and admit its distill job; `created` says which happened.

    `job_id` is null for an origin with no extraction prompt.
    """
    origin = _identifier(origin, "origin")
    external_session_id = _identifier(external_session_id, "external_session_id")
    started = _epoch(started_at, "started_at")
    ended = _epoch(ended_at, "ended_at")
    if ended < started:
        raise ValueError("ended_at must not be earlier than started_at")
    clean_turns = validate_turns(turns)
    metadata_payload = validate_metadata(metadata)
    source_id = conversation_source_id(namespace, origin, external_session_id)
    payload = json.dumps(clean_turns, ensure_ascii=False)
    result = {"id": source_id, "created": False, "turns": len(clean_turns)}
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await namespaces.require_registered(conn, namespace)
            status = await conn.execute(
                f"""
                INSERT INTO "{PG_SCHEMA}".conversation_sources
                  (id, namespace, origin, external_session_id, started_at, ended_at, turns,
                   metadata, created_at, created_by)
                VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb,$9,$10)
                ON CONFLICT (namespace, origin, external_session_id) DO NOTHING
                """,
                source_id,
                namespace,
                origin,
                external_session_id,
                started,
                ended,
                payload,
                metadata_payload,
                time.time(),
                key.label,
            )
            if status.endswith(" 1"):
                job_id = await _admit(conn, key, namespace, origin, source_id)
                return {**result, "created": True, "job_id": job_id}
            existing = await conn.fetchrow(
                f"""
                SELECT created_by, turns, distilled_through
                FROM "{PG_SCHEMA}".conversation_sources
                WHERE id = $1
                FOR UPDATE
                """,
                source_id,
            )
            if not key.is_admin and existing["created_by"] != key.label:
                raise ConversationForbidden(
                    "only the key that stored this conversation source or an admin can replace it"
                )
            stored = existing["turns"]
            if isinstance(stored, str):
                stored = json.loads(stored)
            if stored != clean_turns:
                if clean_turns[: len(stored)] != stored:
                    await _refuse_rewrite(conn, source_id, existing["distilled_through"])
                await conn.execute(
                    f"""
                    UPDATE "{PG_SCHEMA}".conversation_sources
                    SET turns = $2::jsonb, started_at = $3, ended_at = $4, metadata = $5::jsonb
                    WHERE id = $1
                    """,
                    source_id,
                    payload,
                    started,
                    ended,
                    metadata_payload,
                )
            job_id = await _admit(conn, key, namespace, origin, source_id)
    return {**result, "job_id": job_id}


async def _refuse_rewrite(conn: Any, source_id: str, distilled_through: int) -> None:
    """A rewrite that is not an append is refused once notes or the distill cursor depend on it."""
    if distilled_through > 0:
        raise ConversationConflict(
            f"conversation source {source_id} has {distilled_through} distilled turns, so its "
            "stored turns cannot change; re-upload them unchanged with any new turns appended, "
            "or store the changed session under a new external_session_id"
        )
    # A separate statement after the row lock sees notes committed while it waited.
    referenced = await conn.fetchval(
        f"""
        SELECT EXISTS(
          SELECT 1 FROM "{PG_SCHEMA}".memory_chunks WHERE conversation_id = $1
        )
        """,
        source_id,
    )
    if referenced:
        raise ConversationConflict(
            f"conversation source {source_id} is referenced by stored notes, so its stored "
            "turns cannot change; re-upload them unchanged with any new turns appended, or "
            "store the changed session under a new external_session_id"
        )


def _turn_bound(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


async def get_conversation(
    conversation_id: str,
    key: Any,
    *,
    turn_start: Any = None,
    turn_end: Any = None,
) -> dict[str, Any]:
    """A readable source with its turns sliced to [turn_start, turn_end], both inclusive."""
    start = _turn_bound(turn_start, "turn_start")
    end = _turn_bound(turn_end, "turn_end")
    if start is not None and end is not None and start > end:
        raise ValueError("turn_start must not be greater than turn_end")
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        row = await conn.fetchrow(
            f"""
            SELECT id, namespace, origin, external_session_id, started_at, ended_at, turns
            FROM "{PG_SCHEMA}".conversation_sources
            WHERE id = $1
            """,
            conversation_id,
        )
    if row is None or not key.permits(row["namespace"]):
        raise ConversationNotFound(f"unknown conversation source: {conversation_id}")
    turns = row["turns"]
    if isinstance(turns, str):
        turns = json.loads(turns)
    last = len(turns) - 1
    first = 0 if start is None else start
    final = last if end is None else end
    if final > last or first > final:
        raise ValueError(
            f"turn range {first}-{final} is outside conversation source "
            f"{conversation_id}'s {len(turns)} turns"
        )
    return {
        "id": row["id"],
        "namespace": row["namespace"],
        "origin": row["origin"],
        "external_session_id": row["external_session_id"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "turns": [{"index": i, **turns[i]} for i in range(first, final + 1)],
    }
