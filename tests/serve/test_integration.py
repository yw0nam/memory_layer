"""Integration tests for search() and the MCP tools against Postgres and the model services.

A small git repository is indexed into the session's throwaway Postgres container once
per module, so code search runs against rows this module wrote; memory rows are seeded
through save_note. The embedder and reranker are the configured live endpoints.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import uuid

import asyncpg
import numpy as np
import pytest

from memory_base.core.config import EMB_DIM, PG_SCHEMA, db_url, vector_literal
from memory_base.core.db import acquire, close_pool, get_pool
from memory_base.retrieval.search import PER_FILE_CAP, search
from memory_base.serve import mcp_server, repos
from memory_base.serve.notes.store import build_note_row, save_note

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("indexed_code")]

_SEED_SOURCE = '''"""Embedding vector search pipeline: chunks are stored as halfvec columns."""

EMB_DIM = 2048  # halfvec(2048) in pgvector


def embedding_vector_search_pipeline(query):
    """Embed the query, then rank halfvec rows by cosine distance."""
    return query
'''


@pytest.fixture(scope="module")
def indexed_code(tmp_path_factory):
    origin = tmp_path_factory.mktemp("seed-origin")
    (origin / "seed.py").write_text(_SEED_SOURCE)
    git_env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t"}
    git_env.update(GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    for args in (["init", "-q", "-b", "main"], ["add", "."], ["commit", "-q", "-m", "seed"]):
        subprocess.run(["git", "-C", str(origin), *args], check=True, env=git_env)
    asyncio.run(repos._run_ingest_job(str(origin), repos.CACHE_ROOT / "seed", None, "test"))


async def _delete_note(note_id: str) -> None:
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', note_id)
    finally:
        await conn.close()


# ---- search() against seeded code_chunks / memory_chunks ------------------


def test_search_code_source_returns_code_hits_with_line_refs():
    hits = asyncio.run(search("halfvec", source="code", rerank=False))
    assert len(hits) >= 1
    assert all(h.source == "code" for h in hits)
    assert all(":L" in h.ref for h in hits)
    assert all(h.rrf > 0 for h in hits)


def test_search_memory_source_returns_memory_hits():
    content = "integration-test pin: zzz_integ_marker 7f3a9b2c"
    note_id = build_note_row(content, "work", ["test"], 1_700_000_000.0)["id"]
    asyncio.run(_delete_note(note_id))
    try:
        asyncio.run(save_note(content, tags=["test"], kind="work"))
        hits = asyncio.run(search(content, source="memory", rerank=False))
        assert len(hits) >= 1
        assert all(h.source == "memory" for h in hits)
    finally:
        asyncio.run(_delete_note(note_id))


def test_notes_with_distinct_session_ids_are_not_capped_by_search():
    marker = f"zzz_integ_notecap_{uuid.uuid4().hex[:8]}"
    contents = [
        f"integration-test note cap pin {marker}: fact number {i}" for i in range(PER_FILE_CAP + 2)
    ]
    note_ids = []
    try:
        for content in contents:
            result = asyncio.run(save_note(content, tags=["test"], kind="work", allow_similar=True))
            note_ids.append(result["id"])
        hits = asyncio.run(search(marker, source="memory", rerank=False))
        assert len(hits) > PER_FILE_CAP
    finally:
        for note_id in note_ids:
            asyncio.run(_delete_note(note_id))


def test_document_chunks_sharing_a_session_id_are_capped_by_search():
    document_id = f"cap-test-doc-{uuid.uuid4().hex[:8]}"
    marker = f"zzz_integ_doccap_{uuid.uuid4().hex[:8]}"
    embedding = vector_literal(np.zeros(EMB_DIM, dtype=np.float16))
    chunk_count = PER_FILE_CAP + 2

    async def seed() -> None:
        async with acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO "{PG_SCHEMA}".memory_chunks
                  (id, source_type, source_ref, chunk_kind, session_id, content_raw,
                   distilled, embedding, ts_last_active, namespace, metadata)
                VALUES ($1, 'document', $2, 'doc', $2, $3, $3, $4::halfvec, $5,
                        'default', $6::jsonb)
                """,
                [
                    (
                        f"doc:{document_id}:{i}",
                        document_id,
                        f"{marker} chunk number {i} about the topic",
                        embedding,
                        1_700_000_000.0,
                        json.dumps({"search_ref": f"{document_id}#chunk-{i}"}),
                    )
                    for i in range(chunk_count)
                ],
            )

    async def clean() -> None:
        async with acquire() as conn:
            await conn.execute(
                f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE session_id = $1',
                document_id,
            )

    asyncio.run(seed())
    try:
        hits = asyncio.run(search(marker, source="memory", rerank=False))
        matching = [h for h in hits if h.meta["session_id"] == document_id]
        assert 0 < len(matching) <= PER_FILE_CAP
    finally:
        asyncio.run(clean())


def test_search_all_source_with_rerank_populates_rerank_score():
    # min_score=0: this test checks rerank_score population, not the relevance floor.
    hits = asyncio.run(
        search("embedding vector search pipeline", source="all", rerank=True, min_score=0)
    )
    assert len(hits) >= 1
    assert any(h.rerank_score is not None for h in hits)


def test_fts_exact_literal_hits_file_containing_it():
    # "halfvec" appears verbatim in the seeded repository.
    hits = asyncio.run(search("halfvec", source="code", rerank=False))
    assert any("halfvec" in h.text.lower() for h in hits)


def test_repeated_searches_keep_pool_backend_count_flat():
    async def _run():
        await get_pool()

        async def backend_count() -> int:
            async with acquire() as conn:
                return await conn.fetchval(
                    """
                    SELECT count(*)
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                      AND usename = current_user
                      AND application_name = current_setting('application_name')
                    """
                )

        before = await backend_count()
        for _ in range(5):
            await search("halfvec", source="code", rerank=False)
            assert await backend_count() == before
        await close_pool()

    asyncio.run(_run())


# ---- MCP in-process tool call ----------------------------------------------


def test_mcp_search_code_tool_real_call_returns_expected_schema(rest_in_process):
    """Calls the real search_code tool in-process via
    mcp.shared.memory.create_connected_server_and_client_session; the proxy
    routes through the in-process REST app, exercising the full
    search.search() pipeline including the reranker.
    """
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            return await client.call_tool("search_code", {"query": "halfvec", "top_k": 3})

    result = asyncio.run(_run())
    assert not result.isError
    payload = result.structuredContent["result"]
    assert len(payload) >= 1
    for item in payload:
        for key in ("source", "ref", "date", "score", "text"):
            assert key in item
        assert item["source"] == "code"
