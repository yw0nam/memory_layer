"""Profile slots: standing text per namespace that clients deliver at session start.

The server calls no model. A scheduled agent reads a slot's source notes and their hash,
then writes the slot: `user` as text it generated from those notes, `work-rules` as a
selection of note ids the server renders verbatim. A write is stored only while the hash
still matches the notes; every version is kept.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.core.secrets import find_secret
from memory_base.serve import namespaces
from memory_base.serve.consolidate import Note, iso, note_from_row

# Covers the slot rules, the rendering, the budgets, and the documented procedure.
PROFILE_VERSION = "1"
SLOTS = ("user", "work-rules")
SLOT_KINDS = {"user": "personal", "work-rules": "work"}
SLOT_FIELDS = {"user": "content", "work-rules": "note_ids"}
DEFAULT_MAX_CHARS = {"user": 1500, "work-rules": 6000}
MIN_MAX_CHARS = 200
MAX_MAX_CHARS = 20000
MAX_NOTE_IDS = 200
DEFAULT_VERSIONS_LIMIT = 20
MAX_VERSIONS_LIMIT = 200

WRITE_FIELDS = frozenset(
    {
        "namespace",
        "slot",
        "source_hash",
        "author",
        "model",
        "dry_run",
        "content",
        "note_ids",
        "max_chars",
    }
)

_TABLE = f'"{PG_SCHEMA}".profiles'

LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended('profile:' || $1 || ':' || $2, 0))"
SOURCES_SQL = f"""
SELECT id, chunk_kind AS kind, ts_last_active, occurred_at, content_raw AS text, metadata
FROM "{PG_SCHEMA}".memory_chunks
WHERE source_type = 'agent_note' AND archived_at IS NULL AND namespace = $1 AND chunk_kind = $2
ORDER BY ts_last_active, id
"""
LATEST_SQL = f"""
SELECT version, content, source_hash, created_at FROM {_TABLE}
WHERE namespace = $1 AND slot = $2
ORDER BY version DESC
LIMIT 1
"""
INSERT_SQL = f"""
INSERT INTO {_TABLE}
  (namespace, slot, version, content, source_ids, source_hash, author, model, created_at)
VALUES ($1, $2, $3, $4, $5::text[], $6, $7, $8, $9)
RETURNING version
"""
SERVED_SQL = f"""
SELECT DISTINCT ON (namespace, slot) namespace, slot, version, content, created_at
FROM {_TABLE}
WHERE $1::text[] IS NULL OR namespace = ANY($1::text[])
ORDER BY namespace, slot, version DESC
"""
VERSIONS_SQL = f"""
SELECT version, content, source_ids, source_hash, author, model, created_at FROM {_TABLE}
WHERE namespace = $1 AND slot = $2
ORDER BY version DESC
LIMIT $3
"""


class RequestError(ValueError):
    """A write request breaks the schema; REST maps it to 400."""


class Stale(Exception):
    """The submitted source hash no longer matches the slot's notes."""

    def __init__(self, source_hash: str) -> None:
        super().__init__("stale")
        self.source_hash = source_hash


class Refused(ValueError):
    """The write fails a content check; nothing is stored. REST maps it to 400."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class Write:
    namespace: str
    slot: str
    source_hash: str
    author: str
    model: str | None
    dry_run: bool
    content: str | None
    note_ids: tuple[str, ...] | None
    max_chars: int


def source_hash(namespace: str, slot: str, notes: Iterable[Note]) -> str:
    """sha256 over the slot's eligible notes, each note's fields, and the profile version."""
    payload = {
        "v": PROFILE_VERSION,
        "slot": slot,
        "namespace": namespace,
        "notes": [
            [
                note.id,
                hashlib.sha256(note.text.encode()).hexdigest(),
                note.kind,
                note.author,
                note.saved,
                note.occurred_at,
                sorted(note.tags),
            ]
            for note in sorted(notes, key=lambda note: note.id)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def render_rules(notes: Mapping[str, Note], ids: Iterable[str]) -> str:
    """One `- ` item per note in the given order; a note's inner lines are indented two spaces."""
    return "\n".join("- " + notes[i].text.replace("\n", "\n  ") for i in ids)


def _text(body: dict[str, Any], name: str) -> str:
    value = body.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"{name} must be a non-blank string")
    return value


def parse_write(body: Any) -> Write:
    """Validate a write request strictly; `content` belongs to `user`, `note_ids` to work-rules."""
    if not isinstance(body, dict):
        raise RequestError("JSON body must be an object")
    unknown = sorted(set(body) - WRITE_FIELDS)
    if unknown:
        raise RequestError(f"unknown field: {', '.join(unknown)}")
    slot = body.get("slot")
    if slot not in SLOTS:
        raise RequestError(f"slot must be one of {SLOTS}")
    for other_slot, field_name in SLOT_FIELDS.items():
        if other_slot != slot and field_name in body:
            raise RequestError(f"{field_name} is allowed only on slot {other_slot}")
    if SLOT_FIELDS[slot] not in body:
        raise RequestError(f"{SLOT_FIELDS[slot]} is required on slot {slot}")
    content = None
    note_ids = None
    if slot == "user":
        content = body["content"]
        if not isinstance(content, str):
            raise RequestError("content must be a string")
    else:
        raw = body["note_ids"]
        if not isinstance(raw, list) or any(not isinstance(i, str) for i in raw):
            raise RequestError("note_ids must be a list of strings")
        if len(raw) > MAX_NOTE_IDS:
            raise RequestError(f"note_ids must hold at most {MAX_NOTE_IDS} ids")
        if len(set(raw)) != len(raw):
            raise RequestError("note_ids repeats an id")
        note_ids = tuple(raw)
    max_chars = body.get("max_chars", DEFAULT_MAX_CHARS[slot])
    if (
        isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or not MIN_MAX_CHARS <= max_chars <= MAX_MAX_CHARS
    ):
        raise RequestError(
            f"max_chars must be an integer between {MIN_MAX_CHARS} and {MAX_MAX_CHARS}"
        )
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise RequestError("model must be a string or null")
    dry_run = body.get("dry_run", False)
    if not isinstance(dry_run, bool):
        raise RequestError("dry_run must be a boolean")
    return Write(
        namespace=_text(body, "namespace"),
        slot=slot,
        source_hash=_text(body, "source_hash"),
        author=_text(body, "author"),
        model=model,
        dry_run=dry_run,
        content=content,
        note_ids=note_ids,
        max_chars=max_chars,
    )


async def read_sources(conn: Any, namespace: str, slot: str) -> list[Note]:
    """The slot's eligible notes: active agent notes of its kind, by save time then id."""
    return [
        note_from_row(row) for row in await conn.fetch(SOURCES_SQL, namespace, SLOT_KINDS[slot])
    ]


def _source_entry(note: Note) -> dict[str, Any]:
    return {
        "id": note.id,
        "kind": note.kind,
        "author": note.author,
        "saved": iso(note.saved),
        "occurred_at": iso(note.occurred_at),
        "tags": list(note.tags),
        "text": note.text,
    }


async def sources(namespace: str, slot: str) -> dict[str, Any]:
    """A slot's eligible notes, their hash, and the current version, from one snapshot."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            notes = await read_sources(conn, namespace, slot)
            latest = await conn.fetchrow(LATEST_SQL, namespace, slot)
    digest = source_hash(namespace, slot, notes)
    current = None
    if latest is not None:
        current = {
            "version": latest["version"],
            "source_hash": latest["source_hash"],
            "created_at": iso(latest["created_at"]),
        }
    return {
        "namespace": namespace,
        "slot": slot,
        "profile_version": PROFILE_VERSION,
        "source_hash": digest,
        "current": current,
        "stale": current is None or current["source_hash"] != digest,
        "notes": [_source_entry(note) for note in notes],
    }


def _content(request: Write, notes: list[Note]) -> tuple[str, list[str]]:
    """The text to store and its source ids, or Refused."""
    if request.slot == "user":
        content = (request.content or "").strip()
        if not content:
            raise Refused("content must be non-blank")
        secret = find_secret(content)
        if secret is not None:
            raise Refused(f"content contains a credential ({secret})")
        if len(content) > request.max_chars:
            raise Refused(
                f"content is {len(content)} characters, over max_chars {request.max_chars}",
                chars=len(content),
                max_chars=request.max_chars,
            )
        return content, [note.id for note in notes]
    eligible = {note.id: note for note in notes}
    ids = request.note_ids or ()
    outside = [i for i in ids if i not in eligible]
    if outside:
        raise Refused(f"note_ids outside the slot's eligible notes: {', '.join(outside)}")
    content = render_rules(eligible, ids)
    if len(content) > request.max_chars:
        raise Refused(
            f"rendered work-rules is {len(content)} characters, over max_chars {request.max_chars}",
            chars=len(content),
            max_chars=request.max_chars,
        )
    return content, list(ids)


async def write(request: Write) -> dict[str, Any]:
    """Check the hash and content under the slot lock, then store the next version.

    Raises Stale, Refused, or NamespaceError with nothing stored.
    """
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await conn.execute(LOCK_SQL, request.namespace, request.slot)
            await namespaces.require_registered(conn, request.namespace)
            notes = await read_sources(conn, request.namespace, request.slot)
            digest = source_hash(request.namespace, request.slot, notes)
            if digest != request.source_hash:
                raise Stale(digest)
            content, source_ids = _content(request, notes)
            latest = await conn.fetchrow(LATEST_SQL, request.namespace, request.slot)
            if (
                latest is not None
                and latest["source_hash"] == digest
                and latest["content"] == content
            ):
                return {"status": "unchanged", "version": latest["version"]}
            if request.dry_run:
                return {"status": "planned", "content": content, "chars": len(content)}
            version = await conn.fetchval(
                INSERT_SQL,
                request.namespace,
                request.slot,
                1 if latest is None else latest["version"] + 1,
                content,
                source_ids,
                digest,
                request.author,
                request.model,
                time.time(),
            )
    return {"status": "written", "version": version, "chars": len(content)}


async def served(scope: list[str] | None) -> list[dict[str, Any]]:
    """The latest version of every slot in scope (None: every namespace), when non-empty."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(SERVED_SQL, scope)
    latest = sorted(
        (row for row in rows if row["content"]),
        key=lambda row: (row["namespace"], SLOTS.index(row["slot"])),
    )
    return [
        {
            "namespace": row["namespace"],
            "slot": row["slot"],
            "version": row["version"],
            "content": row["content"],
            "created_at": iso(row["created_at"]),
        }
        for row in latest
    ]


async def versions(namespace: str, slot: str, limit: int) -> dict[str, Any]:
    """A slot's versions newest first, with their sources and authorship."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(VERSIONS_SQL, namespace, slot, limit)
    return {
        "namespace": namespace,
        "slot": slot,
        "versions": [
            {
                "version": row["version"],
                "content": row["content"],
                "source_ids": list(row["source_ids"]),
                "source_hash": row["source_hash"],
                "author": row["author"],
                "model": row["model"],
                "created_at": iso(row["created_at"]),
            }
            for row in rows
        ],
    }
