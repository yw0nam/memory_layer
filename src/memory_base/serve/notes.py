"""Validation and storage for agent-authored memory notes."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import asyncpg

from memory_base.common import DB_URL, PG_SCHEMA, VllmEmbedder, embed_text
from memory_base.schema import ensure_schema

NOTE_MAX_CHARS = 4000
NOTE_KINDS = ("note", "decision")


def build_note_row(content: str, kind: str, tags: list[str] | None, now: float) -> dict[str, Any]:
    """Validate a note and map it to memory_chunks columns (no embedding)."""
    content = content.strip()
    if not content:
        raise ValueError("content must not be empty")
    if len(content) > NOTE_MAX_CHARS:
        raise ValueError(f"content exceeds {NOTE_MAX_CHARS} chars")
    if kind not in NOTE_KINDS:
        raise ValueError(f"kind must be one of {NOTE_KINDS}")
    note_id = f"note:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
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
        "metadata": {"tags": tags} if tags else {},
    }


async def save_note(
    content: str, kind: str = "note", tags: list[str] | None = None
) -> dict[str, Any]:
    """Validate, embed, and idempotently store an agent-authored memory."""
    row = build_note_row(content, kind, tags, time.time())
    embedding = await embed_text(VllmEmbedder(), row["raw"])
    conn = await asyncpg.connect(DB_URL)
    try:
        await ensure_schema(conn)
        status = await conn.execute(
            f"""
            INSERT INTO "{PG_SCHEMA}".memory_chunks
              (id, source_type, source_ref, chunk_kind, session_id, content_raw,
               distilled, embedding, ts_last_active, idf_score, metadata)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::halfvec,$9,$10,$11::jsonb)
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
            json.dumps(row["metadata"], ensure_ascii=False),
        )
    finally:
        await conn.close()
    return {"id": row["id"], "kind": row["kind"], "stored": status.endswith(" 1")}
