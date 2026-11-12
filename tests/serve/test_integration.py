"""Integration tests for search() and the MCP tools against Postgres and the model services.

Rows are seeded through save_note into the session's throwaway Postgres container. The
embedder and reranker are the configured live endpoints.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import contextmanager

import asyncpg
import pytest

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.core.db import acquire, close_pool, get_pool
from memory_base.retrieval.search import search
from memory_base.serve import mcp_server
from memory_base.serve.notes.store import save_note

pytestmark = pytest.mark.integration


async def _delete_note(note_id: str) -> None:
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', note_id)
    finally:
        await conn.close()


@contextmanager
def seeded_notes(*contents: str):
    """Save the notes, yield their ids, and delete them afterwards."""
    ids: list[str] = []
    try:
        for content in contents:
            result = asyncio.run(
                save_note(content, tags=["test"], kind="work", allow_similar=True, author="natsume")
            )
            ids.append(result["id"])
        yield ids
    finally:
        for note_id in ids:
            asyncio.run(_delete_note(note_id))


# ---- search() against seeded memory_chunks ----------------------------------


def test_search_returns_the_seeded_note():
    marker = f"zzz_integ_marker_{uuid.uuid4().hex[:8]}"
    with seeded_notes(f"integration-test pin: {marker}") as ids:
        hits = asyncio.run(search(marker, rerank=False))
        assert ids[0] in [h.meta["id"] for h in hits]
        assert all(h.rrf > 0 for h in hits)


def test_every_note_matching_the_query_is_returned():
    marker = f"zzz_integ_notecap_{uuid.uuid4().hex[:8]}"
    contents = [f"integration-test note pin {marker}: fact number {i}" for i in range(5)]
    with seeded_notes(*contents) as ids:
        hits = asyncio.run(search(marker, rerank=False))
        assert set(ids) <= {h.meta["id"] for h in hits}


def test_search_with_rerank_populates_rerank_score():
    marker = f"zzz_integ_rerank_{uuid.uuid4().hex[:8]}"
    with seeded_notes(f"integration-test pin: {marker}"):
        # min_score=0: this test checks rerank_score population, not the relevance floor.
        hits = asyncio.run(search(marker, rerank=True, min_score=0))
        assert any(h.rerank_score is not None for h in hits)


def test_fts_exact_literal_hits_note_containing_it():
    marker = f"zzz_integ_literal_{uuid.uuid4().hex[:8]}"
    with seeded_notes(f"integration-test pin: {marker}"):
        hits = asyncio.run(search(marker, rerank=False))
        assert any(marker in h.text for h in hits)


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
            await search("halfvec", rerank=False)
            assert await backend_count() == before
        await close_pool()

    asyncio.run(_run())


# ---- MCP in-process tool call ----------------------------------------------


def test_mcp_search_memory_tool_real_call_returns_expected_schema(rest_in_process):
    """Calls the real search_memory tool in-process via
    mcp.shared.memory.create_connected_server_and_client_session; the proxy
    routes through the in-process REST app, exercising the full
    search.search() pipeline including the reranker.
    """
    from mcp.shared.memory import create_connected_server_and_client_session

    marker = f"zzz_integ_mcp_{uuid.uuid4().hex[:8]}"

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            return await client.call_tool(
                "search_memory", {"query": marker, "top_k": 3, "min_score": 0, "author": "all"}
            )

    with seeded_notes(f"integration-test pin: {marker}"):
        result = asyncio.run(_run())
    assert not result.isError
    payload = result.structuredContent["result"]
    assert len(payload) >= 1
    for item in payload:
        for key in ("id", "kind", "ref", "date", "score", "text"):
            assert key in item
