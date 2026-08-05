"""Tests for serve/api.py's hit serialization and the MCP tool list.

Runs without LLM/DB access: pure-function unit tests use hand-built
search.Hit objects, and the MCP in-process check only lists tools (the
proxy tools perform no I/O until invoked).
"""

from __future__ import annotations

import asyncio

from memory_base.retrieval.search import Hit

from memory_base.serve import api
from memory_base.serve import mcp_server


def _hit(
    source="code",
    ref="a.py:L1-L2",
    text="hello",
    ts=1_700_000_000.0,
    rrf=0.5,
    rerank_score=None,
    meta=None,
):
    return Hit(
        source=source,
        ref=ref,
        text=text,
        ts=ts,
        rrf=rrf,
        rerank_score=rerank_score,
        meta=meta or {},
    )


# ---- api.py pure functions ---------------------------------------------------


def test_hit_to_dict_basic_fields():
    h = _hit(
        source="code", ref="a.py:L1-L2", text="body", ts=1_700_000_000.0, rrf=0.4, rerank_score=0.8
    )
    d = api.hit_to_dict(h)
    assert d["source"] == "code"
    assert d["ref"] == "a.py:L1-L2"
    assert d["score"] == 0.8  # prefers rerank_score over rrf
    assert d["text"] == "body"
    assert "date" in d and len(d["date"]) == 10  # YYYY-MM-DD
    assert "context" not in d


def test_hit_to_dict_falls_back_to_rrf_when_no_rerank_score():
    h = _hit(rrf=0.4, rerank_score=None)
    d = api.hit_to_dict(h)
    assert d["score"] == 0.4


def test_hit_to_dict_truncates_text_and_includes_context():
    h = _hit(text="z" * 5000, meta={"context": "CTX"})
    d = api.hit_to_dict(h)
    assert len(d["text"]) == 2000
    assert d["context"] == "CTX"


def test_hit_to_dict_includes_repo_for_code_hits():
    assert api.hit_to_dict(_hit(meta={"repo": "YUI"}))["repo"] == "YUI"
    assert "repo" not in api.hit_to_dict(_hit())


def test_hit_to_dict_marks_archived_hits():
    d = api.hit_to_dict(_hit(meta={"archived": True}))
    assert d["archived"] is True


def test_hit_to_dict_omits_archived_key_for_live_hits():
    assert "archived" not in api.hit_to_dict(_hit(meta={"archived": False}))
    assert "archived" not in api.hit_to_dict(_hit())


# ---- MCP server in-process tool registration --------------------------------


def test_mcp_server_registers_expected_tools():
    """Verify tools/list exposes the search tools plus save_memory in-process.

    Uses mcp.shared.memory.create_connected_server_and_client_session to spin
    up an in-memory client/server pair (no subprocess, no stdio). Listing
    tools performs no I/O — the proxy tools only reach REST when invoked.
    """
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.list_tools()
            return {t.name for t in result.tools}

    names = asyncio.run(_run())
    assert names == {
        "search",
        "search_code",
        "search_memory",
        "save_memory",
        "ingest_document",
        "ingest_repo",
        "remove_repo",
        "list_repos",
    }
