"""Unit tests for the session record MCP tools: thin proxies over the REST API.

Same harness as tests/serve/test_mcp_decisions.py: a MockTransport behind
``rest_client.client``, the tool functions called directly, no DB, no network.
"""

from __future__ import annotations

import asyncio

import httpx

from memory_base.serve import mcp_server
from memory_base.serve.common import rest_client
from memory_base.serve.sessions import tools

REPO = "repo:github.com/o/r"


def _patch_client(monkeypatch, captured, body):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.raw_path.decode().split("?")[0]
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=body)

    def fake_client():
        return httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(rest_client, "client", fake_client)


def _tools():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as c:
            return {t.name: t for t in (await c.list_tools()).tools}

    return asyncio.run(_run())


def test_list_sessions_sends_only_the_given_filters(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, captured, [])
    asyncio.run(tools.list_sessions(repo=REPO, since="2026-05-01", limit=5))
    assert captured["method"] == "GET" and captured["path"] == "/sessions"
    assert captured["params"] == {"repo": REPO, "since": "2026-05-01", "limit": "5"}


def test_read_session_puts_the_id_in_the_path_and_maps_the_arguments(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, captured, {"session_id": "a/b", "events": []})
    asyncio.run(tools.read_session("a/b", from_seq=2, limit=10, role="user", contains="ab"))
    assert captured["path"] == "/sessions/a%2Fb"
    assert captured["params"] == {"from_seq": "2", "limit": "10", "role": "user", "contains": "ab"}


def test_search_transcript_maps_query_to_q(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, captured, [])
    asyncio.run(tools.search_transcript("스커트", role="assistant", limit=3))
    assert captured["path"] == "/session_events/search"
    assert captured["params"] == {"q": "스커트", "role": "assistant", "limit": "3"}


def test_the_three_tools_are_registered_without_an_author_argument():
    registered = _tools()
    for name in ("list_sessions", "read_session", "search_transcript"):
        assert name in registered, name
        assert "author" not in registered[name].inputSchema["properties"]
    assert registered["read_session"].inputSchema["required"] == ["session_id"]
    assert registered["search_transcript"].inputSchema["required"] == ["query"]


def test_the_descriptions_say_on_demand_own_sessions_never_ranked():
    registered = _tools()
    for name in ("list_sessions", "read_session", "search_transcript"):
        description = " ".join(registered[name].description.split())
        assert "your own" in description, name
        assert "not memory" in description, name
        assert "never ranked" in description, name


def test_read_session_states_the_page_cap_and_how_to_continue():
    description = " ".join(_tools()["read_session"].description.split())
    assert "200,000 characters" in description
    assert "`next_from_seq`" in description
