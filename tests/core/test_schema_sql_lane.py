"""Unit contracts for the notes, messages, namespaces, keys, decisions, and session events
schema."""

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


def test_decisions_are_numbered_per_scope_and_author(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    sql = "\n".join(conn.queries)
    body = _table_body(sql, "decisions")
    assert "scope text NOT NULL CHECK (scope LIKE 'repo:%')" in body
    assert "author text NOT NULL" in body
    assert "number integer NOT NULL" in body
    assert "title text NOT NULL" in body
    assert "decision text NOT NULL" in body
    assert "why text" in body
    assert "links jsonb NOT NULL DEFAULT '[]'" in body
    assert "decided_on date NOT NULL" in body
    assert "status text NOT NULL CHECK (status IN ('active', 'superseded', 'retired'))" in body
    assert "superseded_by integer" in body
    assert "retired_reason text" in body
    assert "created_at timestamptz NOT NULL" in body
    assert "PRIMARY KEY (scope, author, number)" in body
    assert "embedding" not in body
    assert "decisions__" not in sql


def test_the_schema_creates_only_notes_messages_namespaces_keys_decisions_and_session_events(
    monkeypatch,
):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    tables = re.findall(
        r"CREATE TABLE IF NOT EXISTS \"scratch_schema\"\.(\w+)", "\n".join(conn.queries)
    )
    assert tables == [
        "memory_chunks",
        "messages",
        "namespaces",
        "api_keys",
        "decisions",
        "session_events",
    ]


def test_session_events_are_numbered_per_author_and_session_with_a_trigram_index(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    sql = "\n".join(conn.queries)
    body = _table_body(sql, "session_events")
    assert (
        "role text NOT NULL CHECK (role IN ('user', 'assistant', 'tool_call', 'tool_result'))"
        in body
    )
    assert "UNIQUE (author, session_id, seq)" in body
    assert "PRIMARY KEY (author, id)" in body
    assert "namespace" not in body
    assert "embedding" not in body
    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm" in sql
    assert "USING GIN (content gin_trgm_ops)" in " ".join(sql.split())
