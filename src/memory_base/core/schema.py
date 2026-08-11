"""Public database schema setup for memory chunks and retrieval logging."""

from __future__ import annotations

import asyncio

import asyncpg

from memory_base.core.config import EMB_DIM, PG_SCHEMA, require_env

TABLES_QUERY_ROLE = "memory_tables_query"


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


async def _provision_tables_query_role(conn: asyncpg.Connection) -> None:
    password = _sql_literal(require_env("TABLES_QUERY_PASSWORD"))
    advisory_functions = (
        "pg_advisory_lock(bigint)",
        "pg_advisory_lock(integer,integer)",
        "pg_advisory_lock_shared(bigint)",
        "pg_advisory_lock_shared(integer,integer)",
        "pg_advisory_xact_lock(bigint)",
        "pg_advisory_xact_lock(integer,integer)",
        "pg_advisory_xact_lock_shared(bigint)",
        "pg_advisory_xact_lock_shared(integer,integer)",
        "pg_try_advisory_lock(bigint)",
        "pg_try_advisory_lock(integer,integer)",
        "pg_try_advisory_lock_shared(bigint)",
        "pg_try_advisory_lock_shared(integer,integer)",
        "pg_try_advisory_xact_lock(bigint)",
        "pg_try_advisory_xact_lock(integer,integer)",
        "pg_try_advisory_xact_lock_shared(bigint)",
        "pg_try_advisory_xact_lock_shared(integer,integer)",
    )
    advisory_revokes = "\n".join(
        f"REVOKE EXECUTE ON FUNCTION pg_catalog.{function} FROM PUBLIC;\n"
        f"REVOKE EXECUTE ON FUNCTION pg_catalog.{function} FROM {TABLES_QUERY_ROLE};"
        for function in advisory_functions
    )
    await conn.execute(
        f"""
        DO $role$
        BEGIN
          IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{TABLES_QUERY_ROLE}') THEN
            CREATE ROLE {TABLES_QUERY_ROLE};
          END IF;
        END
        $role$;
        ALTER ROLE {TABLES_QUERY_ROLE} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION
          NOBYPASSRLS NOINHERIT PASSWORD {password};
        ALTER ROLE {TABLES_QUERY_ROLE} RESET ALL;
        REASSIGN OWNED BY {TABLES_QUERY_ROLE} TO memory;
        DROP OWNED BY {TABLES_QUERY_ROLE};
        DO $memberships$
        DECLARE membership record;
        BEGIN
          FOR membership IN
            SELECT granted.rolname
            FROM pg_auth_members AS member
            JOIN pg_roles AS granted ON granted.oid = member.roleid
            WHERE member.member = '{TABLES_QUERY_ROLE}'::regrole
          LOOP
            EXECUTE format('REVOKE %I FROM {TABLES_QUERY_ROLE}', membership.rolname);
          END LOOP;
        END
        $memberships$;
        REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA memory FROM {TABLES_QUERY_ROLE};
        REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA memory FROM {TABLES_QUERY_ROLE};
        REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA memory FROM {TABLES_QUERY_ROLE};
        GRANT USAGE ON SCHEMA memory TO {TABLES_QUERY_ROLE};
        GRANT SELECT ON memory.doc_rows TO {TABLES_QUERY_ROLE};
        REVOKE ALL ON SCHEMA public FROM {TABLES_QUERY_ROLE};
        ALTER TABLE memory.doc_rows ENABLE ROW LEVEL SECURITY;
        ALTER TABLE memory.doc_rows FORCE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS doc_rows_namespace ON memory.doc_rows;
        CREATE POLICY doc_rows_namespace ON memory.doc_rows
          FOR SELECT TO {TABLES_QUERY_ROLE}
          USING (namespace = current_setting('app.namespace', true));
        REVOKE EXECUTE ON FUNCTION pg_catalog.set_config(text,text,boolean) FROM PUBLIC;
        REVOKE EXECUTE ON FUNCTION pg_catalog.set_config(text,text,boolean)
          FROM {TABLES_QUERY_ROLE};
        REVOKE EXECUTE ON FUNCTION pg_catalog.pg_notify(text,text) FROM PUBLIC;
        REVOKE EXECUTE ON FUNCTION pg_catalog.pg_notify(text,text) FROM {TABLES_QUERY_ROLE};
        {advisory_revokes}
        REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA public FROM PUBLIC;
        ALTER DEFAULT PRIVILEGES FOR ROLE memory REVOKE EXECUTE ON ROUTINES FROM PUBLIC;
        ALTER ROLE {TABLES_QUERY_ROLE} SET statement_timeout = '10s';
        ALTER ROLE {TABLES_QUERY_ROLE} SET lock_timeout = '2s';
        ALTER ROLE {TABLES_QUERY_ROLE} SET work_mem = '16MB';
        ALTER ROLE {TABLES_QUERY_ROLE} SET temp_file_limit = '256MB';
        ALTER ROLE {TABLES_QUERY_ROLE} SET max_parallel_workers_per_gather = 0;
        ALTER ROLE {TABLES_QUERY_ROLE} SET track_activities = off;
        """
    )


async def ensure_schema(conn: asyncpg.Connection) -> None:
    """Create the memory storage schema objects when they do not exist."""
    schema = f'"{PG_SCHEMA}"'
    await conn.execute(
        f"""
        CREATE SCHEMA IF NOT EXISTS {schema};
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS {schema}.memory_chunks (
          id text PRIMARY KEY, source_type text NOT NULL, source_ref text NOT NULL,
          chunk_kind text NOT NULL, session_id text NOT NULL, content_raw text NOT NULL,
          distilled text, embedding halfvec({EMB_DIM}) NOT NULL,
          ts_last_active double precision NOT NULL, idf_score double precision,
          metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb
        );
        CREATE EXTENSION IF NOT EXISTS pg_textsearch;
        DROP INDEX IF EXISTS {schema}.memory_chunks__fts;
        CREATE INDEX IF NOT EXISTS memory_chunks_bm25 ON {schema}.memory_chunks
          USING bm25(content_raw) WITH (text_config='english');
        CREATE INDEX IF NOT EXISTS memory_chunks__vec ON {schema}.memory_chunks
          USING hnsw (embedding halfvec_cosine_ops);
        CREATE INDEX IF NOT EXISTS memory_chunks__session ON {schema}.memory_chunks (session_id);
        CREATE INDEX IF NOT EXISTS memory_chunks__tags ON {schema}.memory_chunks
          USING GIN ((metadata->'tags'));
        ALTER TABLE {schema}.memory_chunks
          ADD COLUMN IF NOT EXISTS last_hit_at double precision;
        ALTER TABLE {schema}.memory_chunks
          ADD COLUMN IF NOT EXISTS hit_count bigint NOT NULL DEFAULT 0;
        ALTER TABLE {schema}.memory_chunks
          ADD COLUMN IF NOT EXISTS archived_at double precision;
        ALTER TABLE {schema}.memory_chunks
          ADD COLUMN IF NOT EXISTS namespace text NOT NULL DEFAULT 'default';
        CREATE INDEX IF NOT EXISTS memory_chunks__namespace ON {schema}.memory_chunks (namespace);
        CREATE TABLE IF NOT EXISTS {schema}.doc_rows (
          namespace text NOT NULL,
          document_id text NOT NULL,
          row_index int NOT NULL,
          data jsonb NOT NULL,
          PRIMARY KEY (namespace, document_id, row_index)
        );
        CREATE TABLE IF NOT EXISTS {schema}.namespaces (
          name text PRIMARY KEY,
          created_at double precision NOT NULL
        );
        INSERT INTO {schema}.namespaces (name, created_at) VALUES ('default', 0)
          ON CONFLICT (name) DO NOTHING;
        ALTER TABLE {schema}.namespaces
          ADD COLUMN IF NOT EXISTS visibility text NOT NULL DEFAULT 'public'
          CHECK (visibility IN ('public', 'private'));
        ALTER TABLE {schema}.namespaces
          ADD COLUMN IF NOT EXISTS owner text;
        CREATE TABLE IF NOT EXISTS {schema}.api_keys (
          key_hash text PRIMARY KEY,
          label text NOT NULL,
          home text NOT NULL DEFAULT 'default',
          is_admin boolean NOT NULL DEFAULT false,
          created_at timestamptz NOT NULL DEFAULT now(),
          revoked_at timestamptz
        );
        CREATE TABLE IF NOT EXISTS {schema}.retrieval_log (
          id bigserial PRIMARY KEY,
          query text NOT NULL,
          source text NOT NULL,
          hit_ids text[] NOT NULL,
          ts double precision NOT NULL
        );
        CREATE INDEX IF NOT EXISTS retrieval_log__ts ON {schema}.retrieval_log (ts);
        CREATE TABLE IF NOT EXISTS {schema}.jobs (
          job_id text PRIMARY KEY,
          kind text NOT NULL CHECK (kind IN ('document', 'repo')),
          status text NOT NULL DEFAULT 'queued'
            CHECK (status IN ('queued', 'running', 'succeeded', 'no_op', 'failed')),
          key_id text NOT NULL,
          key_label text NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          updated_at timestamptz NOT NULL DEFAULT now(),
          error text,
          namespace text,
          document_id text,
          origin text,
          mode text,
          filename text,
          spool_path text,
          stage text,
          content_hash text,
          chunks_total integer NOT NULL DEFAULT 0,
          chunks_done integer NOT NULL DEFAULT 0,
          chunks_dropped integer NOT NULL DEFAULT 0,
          rows_written integer NOT NULL DEFAULT 0,
          enrichment_retries integer NOT NULL DEFAULT 0,
          name text,
          action text CHECK (action IS NULL OR action IN ('ingest', 'remove')),
          url text,
          branch text,
          CHECK (kind <> 'document' OR
            (namespace IS NOT NULL AND document_id IS NOT NULL AND mode IS NOT NULL
             AND filename IS NOT NULL AND spool_path IS NOT NULL AND stage IS NOT NULL)),
          CHECK (kind <> 'repo' OR (name IS NOT NULL AND action IS NOT NULL)),
          CHECK (kind <> 'repo' OR action <> 'ingest' OR url IS NOT NULL)
        );
        ALTER TABLE {schema}.jobs
          ADD COLUMN IF NOT EXISTS tags text[] NOT NULL DEFAULT '{{}}'::text[];
        CREATE INDEX IF NOT EXISTS jobs__claim
          ON {schema}.jobs (kind, status, key_id, created_at);
        CREATE INDEX IF NOT EXISTS jobs__document_active
          ON {schema}.jobs (namespace, document_id, status) WHERE kind = 'document';
        CREATE INDEX IF NOT EXISTS jobs__document_list
          ON {schema}.jobs (kind, created_at DESC) WHERE kind = 'document';
        CREATE INDEX IF NOT EXISTS jobs__retention
          ON {schema}.jobs (status, updated_at);
        """
    )
    if PG_SCHEMA == "memory":
        await _provision_tables_query_role(conn)


_schema_once_lock = asyncio.Lock()
_prepared_schemas: set[str] = set()


async def ensure_schema_once(conn: asyncpg.Connection) -> None:
    """Run ensure_schema once per process for the current schema, coalescing concurrent callers."""
    schema_name = PG_SCHEMA
    if schema_name in _prepared_schemas:
        return
    async with _schema_once_lock:
        if schema_name in _prepared_schemas:
            return
        await ensure_schema(conn)
        _prepared_schemas.add(schema_name)
