"""Unit contracts for the notes, messages, namespaces, and keys schema."""

from __future__ import annotations

import asyncio
import re

from memory_base.core import schema


class RecordingConnection:
    def __init__(self):
        self.queries: list[str] = []

    async def execute(self, query, *args):
        assert args == ()
        self.queries.append(query)


def _table_body(sql, table):
    body = sql.split(f'CREATE TABLE IF NOT EXISTS "scratch_schema".{table} (', 1)[1]
    return " ".join(body.split(");", 1)[0].split())


def test_api_keys_stamp_their_own_created_at(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    sql = "\n".join(conn.queries)
    body = sql.split('CREATE TABLE IF NOT EXISTS "scratch_schema".api_keys (', 1)[1]
    assert "created_at timestamptz NOT NULL DEFAULT now()" in body.split(");", 1)[0]


def test_messages_table_carries_the_lifecycle_constraints_and_indexes(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    sql = "\n".join(conn.queries)
    body = sql.split('CREATE TABLE IF NOT EXISTS "scratch_schema".messages (', 1)[1]
    body = body.split(");", 1)[0]
    assert "purpose text NOT NULL CHECK (purpose IN ('message', 'handoff'))" in body
    assert "CHECK (purpose <> 'handoff' OR scope IS NOT NULL)" in body
    assert "CHECK (purpose <> 'message' OR scope IS NULL)" in body
    assert "created_at timestamptz NOT NULL," in body
    assert "embedding" not in body
    assert "WHERE claimed_at IS NULL AND cancelled_at IS NULL AND superseded_at IS NULL" in sql
    assert 'ON "scratch_schema".messages (sender_key, idempotency_key)' in " ".join(sql.split())


def test_memory_chunks_keep_occurred_at(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    body = _table_body("\n".join(conn.queries), "memory_chunks")
    assert "occurred_at double precision" in body


def test_pending_message_index_carries_no_expiry_predicate(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    sql = "\n".join(conn.queries)
    index = sql[sql.index("CREATE INDEX IF NOT EXISTS messages__pending") :]
    index = index[: index.index(";")]
    assert "WHERE claimed_at IS NULL AND cancelled_at IS NULL AND superseded_at IS NULL" in index
    assert "expires_at" not in index


def test_the_schema_creates_only_notes_messages_namespaces_and_keys(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    tables = re.findall(
        r"CREATE TABLE IF NOT EXISTS \"scratch_schema\"\.(\w+)", "\n".join(conn.queries)
    )
    assert tables == ["memory_chunks", "messages", "namespaces", "api_keys"]
