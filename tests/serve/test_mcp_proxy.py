"""Unit tests for the rewritten MCP server: an httpx proxy over the REST API.

memory_base.serve.mcp_server keeps FastMCP
and the tool signatures/docstrings, but each tool body is
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
import inspect
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


def test_tool_list_includes_document_ingestion():
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
        "remove_document",
        "list_repos",
    }


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
    assert captured["json"] == {
        "query": "halfvec index",
        "source": "code",
        "top_k": 5,
    }


def test_search_memory_posts_with_source_memory(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.search_memory(query="burst gate", top_k=3))
    assert captured["json"] == {
        "query": "burst gate",
        "source": "memory",
        "top_k": 3,
    }


def test_search_memory_forwards_kind_and_tags(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(
        mcp_server.search_memory(
            query="decision",
            top_k=4,
            kind="decision",
            tags=["infra"],
        )
    )
    assert captured["json"] == {
        "query": "decision",
        "source": "memory",
        "top_k": 4,
        "kind": "decision",
        "tags": ["infra"],
    }


def test_search_code_forwards_repo_filter(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.search_code("marker", repo=["repo_a"]))
    assert captured["json"]["repo"] == ["repo_a"]


def test_search_all_does_not_expose_repo_filter():
    assert "repo" not in inspect.signature(mcp_server.search_all).parameters


def test_search_all_does_not_expose_memory_only_filters():
    params = inspect.signature(mcp_server.search_all).parameters
    assert "kind" not in params
    assert "tags" not in params


def test_search_tools_do_not_expose_include_atoms():
    assert "include_atoms" not in inspect.signature(mcp_server.search_all).parameters
    assert "include_atoms" not in inspect.signature(mcp_server.search_memory).parameters


def test_search_all_forwards_include_archived(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.search_all("query", include_archived=True))
    assert captured["json"]["include_archived"] is True


def test_search_memory_forwards_include_archived(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.search_memory("query", include_archived=True))
    assert captured["json"]["include_archived"] is True


def test_search_400_non_json_body_raises_generic_value_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="upstream proxy error")

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="backend returned 400"):
        asyncio.run(mcp_server.search_code(query="q"))


def test_search_400_json_without_error_key_raises_generic_value_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "bad request"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="backend returned 400"):
        asyncio.run(mcp_server.search_code(query="q"))


def test_search_all_posts_with_source_all(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.search_all(query="anything", top_k=10))
    assert captured["json"] == {
        "query": "anything",
        "source": "all",
        "top_k": 10,
    }


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


def test_ingest_document_429_non_json_body_raises_generic_value_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="backend returned 429"):
        asyncio.run(mcp_server.ingest_document("content", "guide.md"))


def test_ingest_document_posts_text_as_multipart_and_returns_job_reference(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = request.content
        return httpx.Response(
            202,
            json={
                "job_id": "job-1",
                "status": "queued",
                "status_url": "/ingest/jobs/job-1",
            },
        )

    _patch_client(monkeypatch, handler)
    result = asyncio.run(
        mcp_server.ingest_document(
            "# Guide",
            "guide.md",
            document_id="guide",
            origin="mcp:test",
            mode="force",
        )
    )
    assert captured["path"] == "/ingest/document"
    assert captured["content_type"].startswith("multipart/form-data")
    assert b"# Guide" in captured["body"]
    assert b'name="document_id"' in captured["body"]
    assert result == {"job_id": "job-1", "status_url": "/ingest/jobs/job-1"}


def test_ingest_document_forwards_tags_as_repeated_form_fields(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            202,
            json={"job_id": "job-1", "status": "queued", "status_url": "/ingest/jobs/job-1"},
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.ingest_document("# Guide", "guide.md", tags=["zx bank", "policy"]))
    assert captured["body"].count(b'name="tags"') == 2
    assert b"zx bank" in captured["body"]
    assert b"policy" in captured["body"]


def test_ingest_document_omitted_tags_send_no_tags_field(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            202,
            json={"job_id": "job-1", "status": "queued", "status_url": "/ingest/jobs/job-1"},
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.ingest_document("# Guide", "guide.md"))
    assert b'name="tags"' not in captured["body"]


@pytest.mark.parametrize("filename", ["guide.pdf", "slides.pptx", "table.csv"])
def test_ingest_document_mcp_rejects_binary_formats(filename):
    with pytest.raises(ValueError, match="text formats only"):
        asyncio.run(mcp_server.ingest_document("content", filename))


# ---- remove_document proxying ----------------------------------------------


def test_remove_document_deletes_with_default_namespace(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200, json={"document_id": "guide.md", "namespace": "default", "deleted": 3}
        )

    _patch_client(monkeypatch, handler)
    result = asyncio.run(mcp_server.remove_document("guide.md"))
    assert captured["method"] == "DELETE"
    assert captured["path"] == "/ingest/documents/guide.md"
    assert captured["params"] == {}
    assert result == {"document_id": "guide.md", "namespace": "default", "deleted": 3}


def test_remove_document_forwards_namespace(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200, json={"document_id": "guide.md", "namespace": "team-a", "deleted": 1}
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.remove_document("guide.md", namespace="team-a"))
    assert captured["params"] == {"namespace": "team-a"}


def test_remove_document_403_raises_backend_error_message(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"error": "only the document owner or an admin can delete this document"}
        )

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="only the document owner or an admin"):
        asyncio.run(mcp_server.remove_document("guide.md"))


def test_remove_document_404_raises_backend_error_message(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "document not found"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="document not found"):
        asyncio.run(mcp_server.remove_document("ghost.md"))
