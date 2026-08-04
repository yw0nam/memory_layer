"""Unit tests for the process-lifetime schema setup guard (no DB needed)."""

from __future__ import annotations

import asyncio

import pytest

from memory_base.core import schema


class FakeConnection:
    """Stub connection that only records executed statements."""

    def __init__(self):
        self.calls = 0

    async def execute(self, query, *args):
        self.calls += 1


@pytest.fixture(autouse=True)
def _reset_once_state(monkeypatch):
    """Isolate the module-level once-guard state and PG_SCHEMA across tests."""
    monkeypatch.setattr(schema, "_prepared_schemas", set())
    monkeypatch.setattr(schema, "PG_SCHEMA", "test_schema")


def test_ensure_schema_once_runs_ddl_once_for_repeated_calls(monkeypatch):
    calls = 0

    async def fake_ensure_schema(conn):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(schema, "ensure_schema", fake_ensure_schema)
    conn = FakeConnection()

    asyncio.run(schema.ensure_schema_once(conn))
    asyncio.run(schema.ensure_schema_once(conn))

    assert calls == 1


def test_ensure_schema_once_reruns_after_schema_change(monkeypatch):
    calls = 0

    async def fake_ensure_schema(conn):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(schema, "ensure_schema", fake_ensure_schema)
    conn = FakeConnection()

    asyncio.run(schema.ensure_schema_once(conn))
    monkeypatch.setattr(schema, "PG_SCHEMA", "other_schema")
    asyncio.run(schema.ensure_schema_once(conn))

    assert calls == 2


def test_ensure_schema_once_coalesces_concurrent_callers(monkeypatch):
    calls = 0

    async def fake_ensure_schema(conn):
        nonlocal calls
        await asyncio.sleep(0)  # yield so concurrent callers interleave before the flag is set
        calls += 1

    monkeypatch.setattr(schema, "ensure_schema", fake_ensure_schema)
    conn = FakeConnection()

    async def run_concurrent():
        await asyncio.gather(*(schema.ensure_schema_once(conn) for _ in range(10)))

    asyncio.run(run_concurrent())

    assert calls == 1


def test_ensure_schema_itself_runs_every_time_called_directly():
    conn = FakeConnection()

    asyncio.run(schema.ensure_schema(conn))
    asyncio.run(schema.ensure_schema(conn))

    assert conn.calls == 2


def test_ensure_schema_adds_namespace_column_index_and_registry(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "test_schema")
    captured = {}

    class RecordingConnection(FakeConnection):
        async def execute(self, query, *args):
            captured["sql"] = query
            await super().execute(query, *args)

    conn = RecordingConnection()
    asyncio.run(schema.ensure_schema(conn))
    sql = captured["sql"]
    assert (
        'ALTER TABLE "test_schema".memory_chunks\n'
        "          ADD COLUMN IF NOT EXISTS namespace text NOT NULL DEFAULT 'default';" in sql
    )
    assert 'memory_chunks__namespace ON "test_schema".memory_chunks (namespace)' in sql
    assert 'CREATE TABLE IF NOT EXISTS "test_schema".namespaces' in sql
    assert "name text PRIMARY KEY" in sql
    assert "created_at double precision NOT NULL" in sql
    assert "INSERT INTO \"test_schema\".namespaces (name, created_at) VALUES ('default', 0)" in sql
    assert "ON CONFLICT (name) DO NOTHING;" in sql


def test_ensure_schema_adds_api_keys_table_and_namespace_visibility(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "test_schema")
    captured = {}

    class RecordingConnection(FakeConnection):
        async def execute(self, query, *args):
            captured["sql"] = query
            await super().execute(query, *args)

    conn = RecordingConnection()
    asyncio.run(schema.ensure_schema(conn))
    sql = captured["sql"]
    assert 'CREATE TABLE IF NOT EXISTS "test_schema".api_keys' in sql
    assert "key_hash text PRIMARY KEY" in sql
    assert "label text NOT NULL" in sql
    assert "home text NOT NULL DEFAULT 'default'" in sql
    assert "is_admin boolean NOT NULL DEFAULT false" in sql
    assert "revoked_at timestamptz" in sql
    assert (
        "ADD COLUMN IF NOT EXISTS visibility text NOT NULL DEFAULT 'public'\n"
        "          CHECK (visibility IN ('public', 'private'));" in sql
    )
    assert "ADD COLUMN IF NOT EXISTS owner text;" in sql


def test_ensure_schema_adds_the_shared_jobs_table(monkeypatch):
    captured = {}

    class RecordingConnection(FakeConnection):
        async def execute(self, query, *args):
            captured["sql"] = query
            await super().execute(query, *args)

    asyncio.run(schema.ensure_schema(RecordingConnection()))
    sql = captured["sql"]
    assert 'CREATE TABLE IF NOT EXISTS "test_schema".jobs' in sql
    for column in (
        "job_id text PRIMARY KEY",
        "kind text NOT NULL",
        "key_id text NOT NULL",
        "filename text",
        "spool_path text",
        "url text",
        "branch text",
        "enrichment_retries integer",
    ):
        assert column in sql


def test_rebinding_module_pg_schema_keeps_ddl_and_guard_in_sync(monkeypatch):
    """Rebinding schema.PG_SCHEMA, as eval/retrieval.py's _set_schema does, must move
    both the DDL target and the once-guard's recorded name together."""
    captured_query = {}
    real_execute = FakeConnection.execute

    class RecordingConnection(FakeConnection):
        async def execute(self, query, *args):
            captured_query["sql"] = query
            await real_execute(self, query, *args)

    monkeypatch.setattr(schema, "PG_SCHEMA", "eval_rebound_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema_once(conn))

    assert '"eval_rebound_schema".memory_chunks' in captured_query["sql"]
    assert (
        'CREATE INDEX IF NOT EXISTS retrieval_log__ts ON "eval_rebound_schema".retrieval_log (ts)'
        in captured_query["sql"]
    )
    assert "eval_rebound_schema" in schema._prepared_schemas
