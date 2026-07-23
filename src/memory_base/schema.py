"""Public database schema setup for memory chunks and retrieval logging."""

from __future__ import annotations

import asyncpg

from memory_base.common import PG_SCHEMA


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
        CREATE INDEX IF NOT EXISTS memory_chunks__fts ON {schema}.memory_chunks
          USING GIN (to_tsvector('simple', content_raw));
        CREATE INDEX IF NOT EXISTS memory_chunks__vec ON {schema}.memory_chunks
          USING hnsw (embedding halfvec_cosine_ops);
        CREATE INDEX IF NOT EXISTS memory_chunks__session ON {schema}.memory_chunks (session_id);
        DROP TABLE IF EXISTS {schema}.ingest_state;
        DROP TABLE IF EXISTS {schema}.df_stats;
        DROP TABLE IF EXISTS {schema}.history_session_tokens;
        DROP TABLE IF EXISTS {schema}.history_file_sessions;
        ALTER TABLE {schema}.memory_chunks
          ADD COLUMN IF NOT EXISTS last_hit_at double precision;
        ALTER TABLE {schema}.memory_chunks
          ADD COLUMN IF NOT EXISTS hit_count bigint NOT NULL DEFAULT 0;
        CREATE TABLE IF NOT EXISTS {schema}.retrieval_log (
          id bigserial PRIMARY KEY,
          query text NOT NULL,
          source text NOT NULL,
          hit_ids text[] NOT NULL,
          ts double precision NOT NULL
        );
        """
    )
