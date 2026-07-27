"""Integration tests against the live DB (postgres:5439) and vLLM services
(embedder/reranker/LLM) configured via .env.

Exercises search() and MCP tools against real code_chunks and memory_chunks.
Tests that require a memory_chunks row seed it through save_note and remove
it in a finally block. If the DB is unreachable, the whole module is skipped
so this stays CI-safe.
"""

from __future__ import annotations

import asyncio

import pytest

import asyncpg

from memory_base.common import DB_URL, PG_SCHEMA
from memory_base.serve.notes import build_note_row, save_note


def _db_reachable() -> bool:
    async def _check() -> None:
        conn = await asyncpg.connect(DB_URL, timeout=5)
        await conn.close()

    try:
        asyncio.run(_check())
        return True
    except Exception:
        return False


if not _db_reachable():
    pytest.skip(
        f"DB not reachable at {DB_URL}; skipping integration tests", allow_module_level=True
    )

pytestmark = pytest.mark.integration

from memory_base.serve import mcp_server  # noqa: E402
from memory_base.retrieval.search import search  # noqa: E402


async def _delete_note(note_id: str) -> None:
    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', note_id)
    finally:
        await conn.close()


# ---- search() against real code_chunks / memory_chunks --------------------


def test_search_code_source_returns_code_hits_with_line_refs():
    hits = asyncio.run(search("halfvec", source="code", rerank=False))
    assert len(hits) >= 1
    assert all(h.source == "code" for h in hits)
    assert all(":L" in h.ref for h in hits)
    assert all(h.rrf > 0 for h in hits)


def test_search_memory_source_returns_memory_hits():
    content = "integration-test pin: zzz_integ_marker 7f3a9b2c"
    note_id = build_note_row(content, "note", None, 1_700_000_000.0)["id"]
    asyncio.run(_delete_note(note_id))
    try:
        asyncio.run(save_note(content))
        hits = asyncio.run(search(content, source="memory", rerank=False))
        assert len(hits) >= 1
        assert all(h.source == "memory" for h in hits)
    finally:
        asyncio.run(_delete_note(note_id))


def test_search_all_source_with_rerank_populates_rerank_score():
    hits = asyncio.run(search("embedding vector search pipeline", source="all", rerank=True))
    assert len(hits) >= 1
    assert any(h.rerank_score is not None for h in hits)


def test_fts_exact_literal_hits_file_containing_it():
    # "halfvec" is a literal known to appear verbatim in src/common.py and
    # the spec docs indexed into code_chunks.
    hits = asyncio.run(search("halfvec", source="code", rerank=False))
    assert any("halfvec" in h.text.lower() for h in hits)


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
