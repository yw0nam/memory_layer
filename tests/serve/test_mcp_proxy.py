"""Unit tests for the MCP server as an httpx proxy over the REST API.

Each tool body is an httpx call to REST_URL, never the DB/search pipeline. Tests inject a
mocked transport by monkeypatching ``rest_client.client``, a zero-arg factory returning an
``httpx.AsyncClient(base_url=REST_URL, ...)`` (no real network, no DB).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from memory_base.serve.common import rest_client
from memory_base.serve.notes import tools as note_tools
from memory_base.serve.search import tools as search_tools


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(rest_client, "client", fake_client)


# ---- search proxying --------------------------------------------------------


def test_search_memory_posts_the_query_and_top_k(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory(query="burst gate", top_k=3))
    assert captured["json"] == {
        "query": "burst gate",
        "top_k": 3,
    }


def test_search_memory_forwards_kind_and_tags(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(
        search_tools.search_memory(
            query="decision",
            top_k=4,
            kind="work",
            tags=["infra"],
        )
    )
    assert captured["json"] == {
        "query": "decision",
        "top_k": 4,
        "kind": "work",
        "tags": ["infra"],
    }


def test_search_memory_forwards_since_and_until(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory("last week", since="2026-08-01", until="2026-08-12"))
    assert captured["json"]["since"] == "2026-08-01"
    assert captured["json"]["until"] == "2026-08-12"


# ---- list_notes proxying ----------------------------------------------------


def test_list_notes_gets_notes_with_repeated_params(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = request.url.params.multi_items()
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(
        note_tools.list_notes(
            tags=["infra", "db"],
            kind="work",
            since="2026-08-01",
            until="2026-08-12",
            include_archived=True,
            namespace="team-a",
            limit=5,
        )
    )
    assert captured["method"] == "GET"
    assert captured["path"] == "/notes"
    assert sorted(captured["params"]) == sorted(
        [
            ("tags", "infra"),
            ("tags", "db"),
            ("kind", "work"),
            ("since", "2026-08-01"),
            ("until", "2026-08-12"),
            ("include_archived", "true"),
            ("namespace", "team-a"),
            ("limit", "5"),
        ]
    )


def test_search_memory_forwards_the_author_filter(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory("who decided", author="natsume"))
    assert captured["json"]["author"] == "natsume"


def test_list_notes_forwards_the_author_filter(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = request.url.params.multi_items()
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.list_notes(author=["natsume", "claude-code"]))
    assert captured["params"] == [("author", "natsume"), ("author", "claude-code")]


def test_list_notes_omits_unset_filters(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = request.url.params.multi_items()
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.list_notes())
    assert captured["params"] == []


def test_search_memory_forwards_include_archived(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory("query", include_archived=True))
    assert captured["json"]["include_archived"] is True


def test_search_memory_forwards_min_score(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory(query="burst gate", min_score=0.3))
    assert captured["json"]["min_score"] == 0.3


def test_search_memory_forwards_budget_tokens(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory(query="burst gate", budget_tokens=4000))
    assert captured["json"]["budget_tokens"] == 4000


def test_a_json_error_without_an_error_key_names_the_status(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "bad request"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="backend returned 400"):
        asyncio.run(search_tools.search_memory(query="q"))


def test_search_returns_rest_response_body_unmodified(monkeypatch):
    hits = [{"ref": "save_memory", "date": "2026-01-01", "score": 0.9, "text": "x"}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=hits)

    _patch_client(monkeypatch, handler)
    result = asyncio.run(search_tools.search_memory(query="q", top_k=5))
    assert result == hits


# ---- lifecycle tool proxying ------------------------------------------------


def test_archive_notes_posts_ids_and_the_auth_headers(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        captured["key"] = request.headers.get("x-api-key")
        captured["author"] = request.headers.get("x-author")
        return httpx.Response(200, json={"rows": []})

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("MEMORY_API_KEY", "env-key")
    monkeypatch.setenv("MEMORY_AUTHOR", "natsume")
    asyncio.run(note_tools.archive_notes(["note:a"]))
    assert captured["method"] == "POST"
    assert captured["path"] == "/admin/archive"
    assert captured["json"] == {"ids": ["note:a"]}
    assert captured["key"] == "env-key"
    assert captured["author"] == "natsume"


def test_archive_notes_sends_confirm_only_when_true(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"archived": 1})

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.archive_notes(["note:a"], confirm=True))
    assert captured["json"]["confirm"] is True


def test_delete_notes_posts_ids(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"rows": []})

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.delete_notes(["note:a"]))
    assert captured["path"] == "/admin/notes/delete"
    assert captured["json"] == {"ids": ["note:a"]}


def test_delete_notes_sends_confirm_only_when_true(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"deleted": 1})

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.delete_notes(["note:a"], confirm=True))
    assert captured["json"] == {"ids": ["note:a"], "confirm": True}
