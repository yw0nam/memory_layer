"""Unit tests for the decision-lane MCP tools: thin proxies over the REST API.

Same harness as tests/serve/test_mcp_messages.py: a MockTransport behind
``rest_client.client``, the tool functions called directly, no DB, no network.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from memory_base.serve import mcp_server
from memory_base.serve.common import rest_client
from memory_base.serve.decisions import tools

SCOPE = "repo:github.com/o/r"


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(rest_client, "client", fake_client)


def _capturing(captured, body=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        captured["json"] = json.loads(request.content) if request.content else None
        return httpx.Response(200, json=body if body is not None else {"number": 1})

    return handler


def _tools():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as c:
            return {t.name: t for t in (await c.list_tools()).tools}

    return asyncio.run(_run())


def test_record_decision_posts_only_the_given_fields(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, _capturing(captured))
    asyncio.run(
        tools.record_decision(SCOPE, "Use Postgres", "Postgres holds it.", decided_on="2026-05-01")
    )
    assert captured["method"] == "POST" and captured["path"] == "/decisions"
    assert captured["json"] == {
        "scope": SCOPE,
        "title": "Use Postgres",
        "decision": "Postgres holds it.",
        "decided_on": "2026-05-01",
    }


def test_record_decision_forwards_why_links_and_supersedes(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, _capturing(captured))
    asyncio.run(
        tools.record_decision(SCOPE, "t", "d", why="w", links=["https://x.y/1"], supersedes=2)
    )
    assert captured["json"]["why"] == "w"
    assert captured["json"]["links"] == ["https://x.y/1"]
    assert captured["json"]["supersedes"] == 2


def test_retire_decision_posts_scope_number_and_reason(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, _capturing(captured))
    asyncio.run(tools.retire_decision(SCOPE, 3, "obsolete"))
    assert captured["path"] == "/decisions/retire"
    assert captured["json"] == {"scope": SCOPE, "number": 3, "reason": "obsolete"}


def test_list_decisions_asks_for_active_unless_inactive_is_included(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, _capturing(captured, body=[]))
    asyncio.run(tools.list_decisions(SCOPE))
    assert captured["method"] == "GET" and captured["path"] == "/decisions"
    assert captured["params"] == {"scope": SCOPE, "status": "active"}
    asyncio.run(tools.list_decisions(SCOPE, include_inactive=True))
    assert captured["params"]["status"] == "all"


def test_get_decision_gets_the_number_under_the_scope(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, _capturing(captured))
    asyncio.run(tools.get_decision(SCOPE, 4))
    assert captured["path"] == "/decisions/4"
    assert captured["params"] == {"scope": SCOPE}


def test_the_four_tools_are_registered_without_an_author_argument():
    registered = _tools()
    for name in ("record_decision", "retire_decision", "list_decisions", "get_decision"):
        assert name in registered, name
        assert "author" not in registered[name].inputSchema["properties"]
    assert set(registered["record_decision"].inputSchema["required"]) == {
        "scope",
        "title",
        "decision",
    }


def test_the_write_tools_say_to_act_only_on_the_owners_word():
    registered = _tools()
    for name in ("record_decision", "retire_decision"):
        assert "only when the owner asks" in registered[name].description, name
    record = registered["record_decision"].description
    assert "git remote get-url origin" in record
    assert "local date" in record
    assert "supersede" in record
