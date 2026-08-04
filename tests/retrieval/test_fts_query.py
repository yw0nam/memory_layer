from __future__ import annotations

import asyncio
import inspect
import uuid

import numpy as np
import pytest

from memory_base.core import db
from memory_base.core.config import EMB_DIM, PG_SCHEMA, vector_literal
from memory_base.retrieval import decompose, search

pytestmark = pytest.mark.integration


def _tsquery_sql() -> str:
    return getattr(search, "FTS_TSQUERY_SQL", "websearch_to_tsquery('simple', $1)")


@pytest.fixture()
def seeded_namespace():
    namespace = f"fts-test-{uuid.uuid4().hex}"
    rows = {
        "natural": "ZX Bank won three awards",
        "identifier": "Index settings include PGVEC_HNSW_M16 for graph construction.",
        "phrase": "The system combines results with reciprocal rank fusion.",
        "scattered": "Reciprocal methods compare candidates across a long detailed passage where "
        "unrelated concepts occupy many positions before rank is discussed and more filler separates "
        "the final mention of fusion.",
        "filler-a": "Quarterly accounting policies and branch office schedules.",
        "filler-b": "Customer support procedures for routine card replacements.",
        "stopword-noise": "What did we do in the of it? What did they say about it in the end?",
    }
    embedding = vector_literal(np.zeros(EMB_DIM, dtype=np.float16))

    async def seed() -> None:
        async with db.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO "{PG_SCHEMA}".memory_chunks
                  (id, source_type, source_ref, chunk_kind, session_id, content_raw,
                   distilled, embedding, ts_last_active, idf_score, namespace, metadata)
                VALUES ($1, 'agent_note', $2, 'note', $3, $4, $4, $5::halfvec,
                        0, NULL, $3, '{{}}'::jsonb)
                """,
                [
                    (f"{namespace}-{name}", name, namespace, content, embedding)
                    for name, content in rows.items()
                ],
            )

    async def clean() -> None:
        try:
            async with db.acquire() as conn:
                await conn.execute(
                    f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = $1',
                    namespace,
                )
        finally:
            await db.close_pool()

    asyncio.run(seed())
    try:
        yield namespace
    finally:
        asyncio.run(clean())


def _fts_rows(namespace: str, query: str):
    query_text = getattr(search, "fts_query_text", lambda value: value)(query)

    async def fetch():
        async with db.acquire() as conn:
            return await conn.fetch(
                f"""
                SELECT source_ref
                FROM "{PG_SCHEMA}".memory_chunks
                WHERE namespace = $2
                  AND to_tsvector('simple', content_raw) @@ {_tsquery_sql()}
                ORDER BY ts_rank_cd(to_tsvector('simple', content_raw), {_tsquery_sql()}) DESC
                """,
                query_text,
                namespace,
            )

    return asyncio.run(fetch())


def test_natural_language_query_matches_best_chunk(seeded_namespace):
    rows = _fts_rows(seeded_namespace, "What awards did ZX Bank win in 2023?")

    assert rows
    assert rows[0]["source_ref"] == "natural"
    assert "stopword-noise" not in [row["source_ref"] for row in rows]


def test_exact_identifier_lookup_stays_on_top(seeded_namespace):
    rows = _fts_rows(seeded_namespace, "PGVEC_HNSW_M16")

    assert rows[0]["source_ref"] == "identifier"


def test_quoted_phrase_keeps_phrase_semantics(seeded_namespace):
    rows = _fts_rows(seeded_namespace, '"reciprocal rank fusion"')

    assert [row["source_ref"] for row in rows] == ["phrase"]


def test_all_fts_legs_use_shared_tsquery_expression():
    assert "FTS_TSQUERY_SQL" in inspect.getsource(search._search_memory)
    assert "FTS_TSQUERY_SQL" in inspect.getsource(search._search_code)
    assert decompose.FTS_TSQUERY_SQL is search.FTS_TSQUERY_SQL
    assert "FTS_TSQUERY_SQL" in inspect.getsource(decompose._memory_backup)
