"""Integration tests against the live DB (postgres:5439) and vLLM services
(embedder/reranker/LLM) configured via .env.

Read-only: never writes to the DB and never runs history_index.py /
cocoindex update. If the DB is unreachable, the whole module is skipped so
this stays CI-safe.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import asyncpg  # noqa: E402

from common import DB_URL  # noqa: E402


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
    pytest.skip(f"DB not reachable at {DB_URL}; skipping integration tests", allow_module_level=True)

pytestmark = pytest.mark.integration

import answer  # noqa: E402
import mcp_server  # noqa: E402
from search import search  # noqa: E402


# ---- search() against real code_chunks / memory_chunks --------------------


def test_search_code_source_returns_code_hits_with_line_refs():
    hits = asyncio.run(search("halfvec", source="code", rerank=False))
    assert len(hits) >= 1
    assert all(h.source == "code" for h in hits)
    assert all(":L" in h.ref for h in hits)
    assert all(h.rrf > 0 for h in hits)


def test_search_history_source_returns_history_hits():
    # vector KNN alone (no FTS match needed) is enough to surface rows from
    # the small memory_chunks table regardless of query wording.
    hits = asyncio.run(search("이전에 진행한 작업 내용을 알려줘", source="history", rerank=False))
    assert len(hits) >= 1
    assert all(h.source == "history" for h in hits)


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


def test_mcp_search_code_tool_real_call_returns_expected_schema():
    """Calls the real search_code tool in-process (no monkeypatch) via
    mcp.shared.memory.create_connected_server_and_client_session, exercising
    the full search.search() pipeline including the reranker.
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


# ---- LLM: answer.plan (single integration call, loose assertions) ---------


def test_answer_plan_code_question_selects_code_source():
    query = "search.py의 _rrf_fuse 함수는 어디에 정의되어 있어?"
    source, queries = asyncio.run(answer.plan(query))
    assert source in ("code", "all")
    assert query in queries
