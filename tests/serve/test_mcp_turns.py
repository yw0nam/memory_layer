"""Unit tests for the turn lane MCP tools and the save_memory source session: thin REST proxies."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from memory_base.serve.common import rest_client
from memory_base.serve.notes import tools as note_tools
from memory_base.serve.turns import tools


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        seen["json"] = json.loads(request.content) if request.content else None
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=[])

    monkeypatch.setattr(
        rest_client,
        "client",
        lambda: httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        ),
    )
    monkeypatch.setenv("MEMORY_API_KEY", "k")
    return seen


def test_list_sessions_sends_only_the_given_filters(captured):
    asyncio.run(tools.list_sessions(origin="hermes", repo="repo:h/p", limit=5))
    assert (captured["method"], captured["path"]) == ("GET", "/sessions")
    assert captured["params"] == {"origin": "hermes", "repo": "repo:h/p", "limit": "5"}
    assert captured["key"] == "k"


def test_read_session_addresses_the_session_by_path(captured):
    asyncio.run(tools.read_session("claude_code", "s 1", contains="plan", offset=3))
    assert captured["path"] == "/sessions/claude_code/s 1"
    assert captured["params"] == {"contains": "plan", "offset": "3", "limit": "50"}


def test_search_transcript_sends_the_query_as_q(captured):
    asyncio.run(tools.search_transcript("회사", namespace="personal"))
    assert captured["path"] == "/turns/search"
    assert captured["params"] == {"q": "회사", "namespace": "personal", "limit": "20"}


def test_save_memory_sends_the_source_session(captured):
    asyncio.run(
        note_tools.save_memory(
            "c", "natsume", tags=["t"], kind="work", session_id="s-1", origin="hermes"
        )
    )
    assert captured["json"]["source_session"] == {"origin": "hermes", "session_id": "s-1"}


def test_save_memory_needs_both_session_id_and_origin(captured):
    with pytest.raises(ValueError, match="both or neither"):
        asyncio.run(
            note_tools.save_memory("c", "natsume", tags=["t"], kind="work", origin="hermes")
        )
    assert captured == {}
