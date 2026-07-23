"""Unit tests for the rewritten MCP server: an httpx proxy over the REST API.

Per docs/specs/impl_rest_api.md §5, memory_base.serve.mcp_server keeps FastMCP
and the four existing tool signatures/docstrings, but each tool body becomes
an httpx call to REST_URL instead of touching the DB/search pipeline
directly. Tests inject a mocked transport by monkeypatching
``mcp_server._client``, a zero-arg factory returning an
``httpx.AsyncClient(base_url=REST_URL, ...)`` -- the intended patch point for
a client-construction hook (no real network, no DB).

Collection succeeds (mcp_server.py already exists) but every proxy test below
fails today: the module still calls the DB/search pipeline directly and has
no REST_URL constant or _client hook.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from memory_base.serve import mcp_server


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=mcp_server.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(mcp_server, "_client", fake_client)


# ---- tool registration ----------------------------------------------------


def test_tool_list_is_exactly_four_tools():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.list_tools()
            return {t.name for t in result.tools}

    names = asyncio.run(_run())
    assert names == {"search", "search_code", "search_history", "save_memory"}


# ---- search proxying --------------------------------------------------------


def test_search_code_posts_to_search_with_source_code(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.search_code(query="halfvec index", top_k=5))
    assert captured["method"] == "POST"
    assert captured["path"] == "/search"
    assert captured["json"] == {"query": "halfvec index", "source": "code", "top_k": 5}


def test_search_history_posts_with_source_history(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.search_history(query="burst gate", top_k=3))
    assert captured["json"] == {"query": "burst gate", "source": "history", "top_k": 3}


def test_search_all_posts_with_source_all(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.search_all(query="anything", top_k=10))
    assert captured["json"] == {"query": "anything", "source": "all", "top_k": 10}


def test_search_returns_rest_response_body_unmodified(monkeypatch):
    hits = [
        {"source": "code", "ref": "a.py:L1-L2", "date": "2026-01-01", "score": 0.9, "text": "x"}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=hits)

    _patch_client(monkeypatch, handler)
    result = asyncio.run(mcp_server.search_code(query="q", top_k=5))
    assert result == hits


# ---- save_memory proxying ---------------------------------------------------


def test_save_memory_posts_to_save_memory_and_returns_body(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "note:abc", "kind": "note", "stored": True})

    _patch_client(monkeypatch, handler)
    result = asyncio.run(mcp_server.save_memory("distilled content", kind="note", tags=["infra"]))
    assert captured["path"] == "/save_memory"
    assert captured["json"] == {
        "content": "distilled content",
        "kind": "note",
        "tags": ["infra"],
        "supersedes": None,
    }
    assert result == {"id": "note:abc", "kind": "note", "stored": True}


def test_save_memory_400_response_raises_value_error_with_server_message(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "content must not be empty"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="content must not be empty"):
        asyncio.run(mcp_server.save_memory(""))
