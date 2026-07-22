"""Tests for src/answer.py and src/mcp_server.py.

Runs without LLM/DB access: pure-function unit tests use hand-built
search.Hit objects, and the MCP in-process check monkeypatches
search.search so no network/DB call happens.
"""

from __future__ import annotations

import asyncio


from memory_base.retrieval.search import Hit

from memory_base.serve import answer
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


# ---- answer.py pure functions ----------------------------------------------


def test_dedup_sort_hits_dedupes_by_source_ref_and_sorts():
    hits = [
        _hit(source="code", ref="a.py:L1-L2", rrf=0.1, rerank_score=0.3),
        _hit(source="code", ref="a.py:L1-L2", rrf=0.9, rerank_score=0.9),  # dup, better score wins
        _hit(source="code", ref="b.py:L1-L2", rrf=0.5, rerank_score=None),  # no rerank -> uses rrf
        _hit(source="history", ref="sess-1", rrf=0.2, rerank_score=0.1),
    ]
    out = answer.dedup_sort_hits(hits, top_k=10)

    assert len(out) == 3  # dup collapsed
    assert out[0].ref == "a.py:L1-L2"
    assert out[0].rerank_score == 0.9
    # remaining ordered by score desc: b.py (rrf 0.5) then sess-1 (rerank 0.1)
    assert [h.ref for h in out[1:]] == ["b.py:L1-L2", "sess-1"]


def test_dedup_sort_hits_respects_top_k():
    hits = [_hit(ref=f"f{i}.py:L1-L2", rrf=float(i)) for i in range(15)]
    out = answer.dedup_sort_hits(hits, top_k=10)
    assert len(out) == 10
    assert out[0].ref == "f14.py:L1-L2"  # highest rrf first


def test_format_evidence_block_includes_number_source_ref_date_text():
    hits = [_hit(source="code", ref="a.py:L1-L2", text="x" * 10, ts=1_700_000_000.0)]
    block = answer.format_evidence_block(hits)
    assert "[1]" in block
    assert "source=code" in block
    assert "ref=a.py:L1-L2" in block
    assert "date=" in block
    assert "x" * 10 in block


def test_format_evidence_block_truncates_text_to_2000_chars():
    hits = [_hit(text="y" * 5000)]
    block = answer.format_evidence_block(hits)
    assert "y" * 2000 in block
    assert "y" * 2001 not in block


def test_format_evidence_block_includes_code_context():
    hits = [_hit(source="code", meta={"context": "SURROUNDING_CODE"})]
    block = answer.format_evidence_block(hits)
    assert "SURROUNDING_CODE" in block


def test_format_evidence_block_ignores_context_for_history_hits():
    hits = [_hit(source="history", meta={"context": "SHOULD_NOT_APPEAR"})]
    block = answer.format_evidence_block(hits)
    assert "SHOULD_NOT_APPEAR" not in block


def test_format_references_section_maps_numbers_to_refs():
    hits = [
        _hit(ref="a.py:L1-L2"),
        _hit(ref="sess-42"),
    ]
    section = answer.format_references_section(hits)
    assert section.startswith("참조:")
    assert "[1] a.py:L1-L2" in section
    assert "[2] sess-42" in section


def test_answer_returns_no_evidence_message_without_llm_call(monkeypatch):
    async def fake_search(query, source="all", rerank=True):
        return []

    monkeypatch.setattr(answer, "search", fake_search)

    called = False

    def fail_llm_client():
        nonlocal called
        called = True
        raise AssertionError("LLM should not be called when there is no evidence")

    monkeypatch.setattr(answer, "llm_client", fail_llm_client)

    result = asyncio.run(answer.answer("아무 상관없는 질문", source="all"))
    assert result == "관련 증거를 찾지 못했습니다."
    assert not called


# ---- mcp_server.py pure functions -------------------------------------------


def test_hit_to_dict_basic_fields():
    h = _hit(
        source="code", ref="a.py:L1-L2", text="body", ts=1_700_000_000.0, rrf=0.4, rerank_score=0.8
    )
    d = mcp_server.hit_to_dict(h)
    assert d["source"] == "code"
    assert d["ref"] == "a.py:L1-L2"
    assert d["score"] == 0.8  # prefers rerank_score over rrf
    assert d["text"] == "body"
    assert "date" in d and len(d["date"]) == 10  # YYYY-MM-DD
    assert "context" not in d


def test_hit_to_dict_falls_back_to_rrf_when_no_rerank_score():
    h = _hit(rrf=0.4, rerank_score=None)
    d = mcp_server.hit_to_dict(h)
    assert d["score"] == 0.4


def test_hit_to_dict_truncates_text_and_includes_context():
    h = _hit(text="z" * 5000, meta={"context": "CTX"})
    d = mcp_server.hit_to_dict(h)
    assert len(d["text"]) == 2000
    assert d["context"] == "CTX"


# ---- MCP server in-process tool registration --------------------------------


def test_mcp_server_registers_three_search_tools(monkeypatch):
    """Verify tools/list exposes search, search_code, search_history in-process.

    Uses mcp.shared.memory.create_connected_server_and_client_session to spin
    up an in-memory client/server pair (no subprocess, no stdio). We
    monkeypatch mcp_server.run_search (aliased from search.search) so no
    DB/LLM access happens even if a tool were invoked.
    """
    from mcp.shared.memory import create_connected_server_and_client_session

    async def fake_search(query, source="all", rerank=True):
        return []

    monkeypatch.setattr(mcp_server, "run_search", fake_search)

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.list_tools()
            return {t.name for t in result.tools}

    names = asyncio.run(_run())
    assert names == {"search", "search_code", "search_history"}
