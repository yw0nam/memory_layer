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
    index = getattr(search, "MEMORY_BM25_INDEX", "memory_chunks_bm25")

    async def fetch():
        async with db.acquire() as conn:
            return await conn.fetch(
                f"""
                SELECT source_ref
                FROM "{PG_SCHEMA}".memory_chunks
                WHERE namespace = $2
                  AND (content_raw <@> to_bm25query($1, '{index}')) < 0
                ORDER BY content_raw <@> to_bm25query($1, '{index}')
                """,
                query,
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


def test_short_chunk_ranks_above_longer_matching_chunk(seeded_namespace):
    rows = _fts_rows(seeded_namespace, "reciprocal rank fusion")

    refs = [row["source_ref"] for row in rows]
    assert refs[0] == "phrase"
    assert "scattered" in refs


def test_all_fts_legs_use_shared_bm25_indexes():
    assert "MEMORY_BM25_INDEX" in inspect.getsource(search._search_memory)
    assert "CODE_BM25_INDEX" in inspect.getsource(search._search_code)
    assert decompose.MEMORY_BM25_INDEX is search.MEMORY_BM25_INDEX
    assert "MEMORY_BM25_INDEX" in inspect.getsource(decompose._memory_backup)
    assert not hasattr(search, "FTS_TSQUERY_SQL")
    assert not hasattr(search, "fts_query_text")
