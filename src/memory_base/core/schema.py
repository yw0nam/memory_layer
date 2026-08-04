"""Public database schema setup for memory chunks and retrieval logging."""

from __future__ import annotations

import asyncio

import asyncpg

from memory_base.core.config import PG_SCHEMA


async def ensure_schema(conn: asyncpg.Connection) -> None:
    """Create the memory storage schema objects when they do not exist."""
    schema = f'"{PG_SCHEMA}"'
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.memory_chunks (
          id text PRIMARY KEY, source_type text NOT NULL, source_ref text NOT NULL,
          chunk_kind text NOT NULL, session_id text NOT NULL, content_raw text NOT NULL,
          distilled text, embedding halfvec(2048) NOT NULL,
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
