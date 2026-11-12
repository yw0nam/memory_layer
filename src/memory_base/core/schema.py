"""Public database schema setup for notes, messages, namespaces, and keys."""

from __future__ import annotations

import asyncio

import asyncpg

from memory_base.core.config import EMB_DIM, PG_SCHEMA


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
          ts_last_active double precision NOT NULL,
          metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
          archived_at double precision,
          namespace text NOT NULL DEFAULT 'default',
          occurred_at double precision
        );
        CREATE EXTENSION IF NOT EXISTS pg_textsearch;
        CREATE INDEX IF NOT EXISTS memory_chunks_bm25 ON {schema}.memory_chunks
          USING bm25(content_raw) WITH (text_config='english');
        CREATE INDEX IF NOT EXISTS memory_chunks__vec ON {schema}.memory_chunks
          USING hnsw (embedding halfvec_cosine_ops);
        CREATE INDEX IF NOT EXISTS memory_chunks__session ON {schema}.memory_chunks (session_id);
        CREATE INDEX IF NOT EXISTS memory_chunks__tags ON {schema}.memory_chunks
          USING GIN ((metadata->'tags'));
        CREATE INDEX IF NOT EXISTS memory_chunks__namespace ON {schema}.memory_chunks (namespace);
        CREATE TABLE IF NOT EXISTS {schema}.messages (
          id uuid PRIMARY KEY,
          namespace text NOT NULL,
          purpose text NOT NULL CHECK (purpose IN ('message', 'handoff')),
          scope text,
          subject text NOT NULL,
          subject_key text NOT NULL,
          status text NOT NULL CHECK (status IN ('info', 'in_progress', 'blocked', 'completed')),
          content text NOT NULL,
          author text NOT NULL,
          sender_key text NOT NULL,
          idempotency_key text,
          created_at timestamptz NOT NULL,
          claimed_at timestamptz,
          cancelled_at timestamptz,
          superseded_at timestamptz,
          expires_at timestamptz,
          CHECK (purpose <> 'handoff' OR scope IS NOT NULL),
          CHECK (purpose <> 'message' OR scope IS NULL)
        );
        CREATE INDEX IF NOT EXISTS messages__pending
          ON {schema}.messages (namespace, purpose, subject_key, created_at DESC, id DESC)
          WHERE claimed_at IS NULL AND cancelled_at IS NULL AND superseded_at IS NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS messages__idempotency
          ON {schema}.messages (sender_key, idempotency_key)
          WHERE idempotency_key IS NOT NULL;
        CREATE TABLE IF NOT EXISTS {schema}.namespaces (
          name text PRIMARY KEY,
          created_at double precision NOT NULL
        );
        INSERT INTO {schema}.namespaces (name, created_at) VALUES ('default', 0)
          ON CONFLICT (name) DO NOTHING;
        CREATE TABLE IF NOT EXISTS {schema}.api_keys (
          key_hash text PRIMARY KEY,
          label text NOT NULL,
          created_at timestamptz NOT NULL DEFAULT now(),
          revoked_at timestamptz
        );
        """
    )


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
