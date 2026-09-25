"""Unit contracts for doc_rows and the restricted table-query role."""

from __future__ import annotations

import asyncio

from memory_base.core import schema


class RecordingConnection:
    def __init__(self):
        self.queries: list[str] = []

    async def execute(self, query, *args):
        assert args == ()
        self.queries.append(query)


def test_ensure_schema_creates_doc_rows_with_only_its_primary_key(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    sql = "\n".join(conn.queries)
    assert 'CREATE TABLE IF NOT EXISTS "scratch_schema".doc_rows' in sql
    assert "namespace text NOT NULL" in sql
    assert "document_id text NOT NULL" in sql
    assert "row_index int NOT NULL" in sql
    assert "data jsonb NOT NULL" in sql
    assert "PRIMARY KEY (namespace, document_id, row_index)" in sql
    assert "doc_rows__" not in sql


def test_ensure_schema_adds_the_api_key_author_allowlist(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    sql = "\n".join(conn.queries)
    assert 'ALTER TABLE "scratch_schema".api_keys' in sql
    assert "ADD COLUMN IF NOT EXISTS authors text[] NOT NULL DEFAULT '{}'" in sql


def test_api_keys_and_jobs_stamp_their_own_created_at(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "scratch_schema")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    sql = "\n".join(conn.queries)
    for table in ("api_keys", "jobs"):
        body = sql.split(f'CREATE TABLE IF NOT EXISTS "scratch_schema".{table} (', 1)[1]
        body = body.split(");", 1)[0]
        assert "created_at timestamptz NOT NULL DEFAULT now()" in body, table


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


def test_production_schema_self_heals_query_role_and_hardens_function_acls(
    monkeypatch,
):
    monkeypatch.setattr(schema, "PG_SCHEMA", "memory")
    monkeypatch.setenv("TABLES_QUERY_PASSWORD", "quote'me")
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    sql = "\n".join(conn.queries)
    assert "CREATE ROLE memory_tables_query" in sql
    assert (
        "ALTER ROLE memory_tables_query LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
        "NOREPLICATION NOBYPASSRLS NOINHERIT PASSWORD 'quote''me'" in " ".join(sql.split())
    )
    assert "FROM pg_auth_members" in sql
    assert "ALTER ROLE memory_tables_query RESET ALL" in sql
    assert "REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA memory" in sql
    assert "GRANT USAGE ON SCHEMA memory TO memory_tables_query" in sql
    assert "GRANT SELECT ON memory.doc_rows TO memory_tables_query" in sql
    assert "REVOKE ALL ON SCHEMA public FROM memory_tables_query" in sql
    assert "ALTER TABLE memory.doc_rows ENABLE ROW LEVEL SECURITY" in sql
    assert "ALTER TABLE memory.doc_rows FORCE ROW LEVEL SECURITY" in sql
    assert "DROP POLICY IF EXISTS doc_rows_namespace ON memory.doc_rows" in sql
    assert "CREATE POLICY doc_rows_namespace ON memory.doc_rows" in sql
    assert "FOR SELECT TO memory_tables_query" in sql
    assert "namespace = current_setting('app.namespace', true)" in sql
    assert "REVOKE EXECUTE ON FUNCTION pg_catalog.set_config(text,text,boolean) FROM PUBLIC" in sql
    assert "REVOKE EXECUTE ON FUNCTION pg_catalog.pg_notify(text,text) FROM PUBLIC" in sql
    for function in (
        "pg_advisory_lock(bigint)",
        "pg_advisory_lock(integer,integer)",
        "pg_advisory_lock_shared(bigint)",
        "pg_advisory_xact_lock(bigint)",
        "pg_try_advisory_lock(bigint)",
        "pg_try_advisory_xact_lock_shared(integer,integer)",
    ):
        assert f"pg_catalog.{function}" in sql
    assert "pg_advisory_unlock_all" not in sql
    assert "REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA public FROM PUBLIC" in sql
    assert "ALTER DEFAULT PRIVILEGES FOR ROLE memory REVOKE EXECUTE ON ROUTINES FROM PUBLIC" in sql
    for setting in (
        "statement_timeout = '10s'",
        "lock_timeout = '2s'",
        "work_mem = '16MB'",
        "temp_file_limit = '256MB'",
        "max_parallel_workers_per_gather = 0",
        "track_activities = off",
    ):
        assert f"ALTER ROLE memory_tables_query SET {setting}" in sql


def test_scratch_schema_does_not_require_or_retarget_query_role(monkeypatch):
    monkeypatch.setattr(schema, "PG_SCHEMA", "memory_eval_scratch")
    monkeypatch.delenv("TABLES_QUERY_PASSWORD", raising=False)
    conn = RecordingConnection()

    asyncio.run(schema.ensure_schema(conn))

    sql = "\n".join(conn.queries)
    assert '"memory_eval_scratch".doc_rows' in sql
    assert "memory_tables_query" not in sql
