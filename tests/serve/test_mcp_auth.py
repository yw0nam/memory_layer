"""Unit tests for MCP API-key forwarding: X-API-Key from the request context,
falling back to MEMORY_API_KEY for stdio (no HTTP request), and the
namespace argument passing straight through instead of being resolved
against a header-derived allowed set (namespace resolution now lives in
REST, not MCP).

No DB/network: mcp_server._client is mocked via httpx.MockTransport, matching
the convention in tests/serve/test_mcp_proxy.py. The connection's Context is
faked with a minimal object exposing ``.request_context.request.headers``,
mirroring the shape mcp.server.fastmcp.Context exposes over streamable HTTP.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from memory_base.serve import mcp_server


class FakeHeaders:
    def __init__(self, headers: dict[str, str]):
        self._headers = {k.lower(): v for k, v in headers.items()}

    def get(self, key: str, default=None):
        return self._headers.get(key.lower(), default)


class FakeRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = FakeHeaders(headers)


class FakeRequestContext:
    def __init__(self, request: FakeRequest | None):
        self.request = request


class FakeCtx:
    def __init__(self, headers: dict[str, str] | None = None, no_request: bool = False):
        request = None if no_request else FakeRequest(headers or {})
        self.request_context = FakeRequestContext(request)


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=mcp_server.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(mcp_server, "_client", fake_client)


# ---- _api_key: request header vs MEMORY_API_KEY fallback -------------------


def test_api_key_from_request_header():
    ctx = FakeCtx(headers={"X-API-Key": "from-header"})
    assert mcp_server._api_key(ctx) == "from-header"


def test_api_key_header_lookup_is_case_insensitive():
    ctx = FakeCtx(headers={"x-api-key": "from-header"})
    assert mcp_server._api_key(ctx) == "from-header"


def test_api_key_no_ctx_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("MEMORY_API_KEY", "from-env")
    assert mcp_server._api_key(None) == "from-env"


def test_api_key_ctx_with_no_request_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("MEMORY_API_KEY", "from-env")
    ctx = FakeCtx(no_request=True)
    assert mcp_server._api_key(ctx) == "from-env"


def test_api_key_ctx_with_no_header_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("MEMORY_API_KEY", "from-env")
    ctx = FakeCtx(headers={})
    assert mcp_server._api_key(ctx) == "from-env"


def test_api_key_absent_everywhere_is_none(monkeypatch):
    monkeypatch.delenv("MEMORY_API_KEY", raising=False)
    assert mcp_server._api_key(None) is None


def test_auth_headers_omits_key_when_absent(monkeypatch):
    monkeypatch.delenv("MEMORY_API_KEY", raising=False)
    assert mcp_server._auth_headers(None) == {}


def test_auth_headers_includes_key_when_present():
    ctx = FakeCtx(headers={"X-API-Key": "secret"})
    assert mcp_server._auth_headers(ctx) == {"x-api-key": "secret"}


# ---- every REST call forwards the resolved header ---------------------------


def test_search_forwards_api_key_header(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-API-Key": "secret"})
    asyncio.run(mcp_server.search_memory(query="q", ctx=ctx))
    assert captured["header"] == "secret"


def test_save_memory_forwards_api_key_header(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get("x-api-key")
        return httpx.Response(
            200,
            json={
                "id": "note:x",
                "kind": "note",
                "stored": True,
                "superseded": None,
                "similar": [],
            },
        )

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-API-Key": "secret"})
    asyncio.run(mcp_server.save_memory("content", ctx=ctx))
    assert captured["header"] == "secret"


def test_ingest_document_forwards_api_key_header(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get("x-api-key")
        return httpx.Response(
            202, json={"job_id": "job-1", "status": "queued", "status_url": "/ingest/jobs/job-1"}
        )

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-API-Key": "secret"})
    asyncio.run(mcp_server.ingest_document("# Guide", "guide.md", ctx=ctx))
    assert captured["header"] == "secret"


def test_ingest_repo_forwards_api_key_header(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get("x-api-key")
        return httpx.Response(
            202,
            json={
                "job_id": "j1",
                "name": "repo",
                "status": "queued",
                "status_url": "/repos/jobs/j1",
            },
        )

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-API-Key": "secret"})
    asyncio.run(mcp_server.ingest_repo("https://github.com/o/repo.git", ctx=ctx))
    assert captured["header"] == "secret"


def test_remove_repo_forwards_api_key_header(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get("x-api-key")
        return httpx.Response(
            202,
            json={
                "job_id": "j2",
                "name": "repo",
                "status": "queued",
                "status_url": "/repos/jobs/j2",
            },
        )

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-API-Key": "secret"})
    asyncio.run(mcp_server.remove_repo("repo", ctx=ctx))
    assert captured["header"] == "secret"


def test_list_repos_forwards_api_key_header(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["header"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-API-Key": "secret"})
    asyncio.run(mcp_server.list_repos(ctx=ctx))
    assert captured["header"] == "secret"


# ---- namespace argument passes through unresolved ---------------------------


def test_search_omitted_namespace_sends_no_namespaces_key(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.search_memory(query="q"))
    assert "namespaces" not in captured["json"]


def test_search_explicit_namespace_sent_as_single_item_list(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.search_memory(query="q", namespace="team-a"))
    assert captured["json"]["namespaces"] == ["team-a"]


def test_save_memory_omitted_namespace_sends_no_namespace_key(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "note:x",
                "kind": "note",
                "stored": True,
                "superseded": None,
                "similar": [],
            },
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.save_memory("content"))
    assert "namespace" not in captured["json"]


def test_save_memory_explicit_namespace_forwarded(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "note:x",
                "kind": "note",
                "stored": True,
                "superseded": None,
                "similar": [],
            },
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.save_memory("content", namespace="team-a"))
    assert captured["json"]["namespace"] == "team-a"


def test_ingest_document_omitted_namespace_sends_no_namespace_field(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            202, json={"job_id": "job-1", "status": "queued", "status_url": "/ingest/jobs/job-1"}
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.ingest_document("# Guide", "guide.md"))
    assert b'name="namespace"' not in captured["body"]


def test_ingest_document_explicit_namespace_forwarded(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            202, json={"job_id": "job-1", "status": "queued", "status_url": "/ingest/jobs/job-1"}
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.ingest_document("# Guide", "guide.md", namespace="team-a"))
    assert b'name="namespace"' in captured["body"]
    assert b"team-a" in captured["body"]


@pytest.mark.parametrize(
    "tool_call",
    [
        pytest.param(lambda: mcp_server.search_memory(query="q"), id="search"),
        pytest.param(lambda: mcp_server.save_memory("content"), id="save-memory"),
        pytest.param(
            lambda: mcp_server.ingest_document("content", "guide.md"), id="ingest-document"
        ),
        pytest.param(
            lambda: mcp_server.ingest_repo("https://github.com/o/repo.git"), id="ingest-repo"
        ),
        pytest.param(lambda: mcp_server.remove_repo("repo"), id="remove-repo"),
        pytest.param(lambda: mcp_server.list_repos(), id="list-repos"),
    ],
)
def test_tools_preserve_forbidden_backend_message(monkeypatch, tool_call):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "namespace access denied"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="namespace access denied"):
        asyncio.run(tool_call())


def test_search_preserves_unauthorized_backend_message(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid or missing API key"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="invalid or missing API key"):
        asyncio.run(mcp_server.search_memory(query="q"))
