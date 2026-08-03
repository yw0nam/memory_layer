"""Unit tests for MCP namespace resolution: X-Memory-Namespaces header parsing,
allowed-set enforcement, and search=all/write=home defaults.

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


# ---- _allowed_namespaces: header parsing -----------------------------------


def test_no_ctx_means_default_only():
    assert mcp_server._allowed_namespaces(None) == ["default"]


def test_ctx_with_no_request_means_default_only():
    ctx = FakeCtx(no_request=True)
    assert mcp_server._allowed_namespaces(ctx) == ["default"]


def test_ctx_with_no_header_means_default_only():
    ctx = FakeCtx(headers={})
    assert mcp_server._allowed_namespaces(ctx) == ["default"]


def test_header_parses_ordered_comma_separated_list():
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b, team-c"})
    assert mcp_server._allowed_namespaces(ctx) == ["team-a", "team-b", "team-c"]


def test_header_lookup_is_case_insensitive():
    ctx = FakeCtx(headers={"x-memory-namespaces": "team-a"})
    assert mcp_server._allowed_namespaces(ctx) == ["team-a"]


def test_header_strips_whitespace_and_drops_empties():
    ctx = FakeCtx(headers={"X-Memory-Namespaces": " team-a ,, team-b ,"})
    assert mcp_server._allowed_namespaces(ctx) == ["team-a", "team-b"]


def test_blank_header_means_default_only():
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "   "})
    assert mcp_server._allowed_namespaces(ctx) == ["default"]


# ---- _resolve_read_namespaces: search=all default, narrow, reject ---------


def test_read_resolution_omitted_covers_whole_allowed_set():
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    assert mcp_server._resolve_read_namespaces(None, ctx) == ["team-a", "team-b"]


def test_read_resolution_narrows_to_named_entry():
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    assert mcp_server._resolve_read_namespaces("team-b", ctx) == ["team-b"]


def test_read_resolution_rejects_namespace_outside_allowed_set():
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    with pytest.raises(ValueError, match="outside the allowed set"):
        mcp_server._resolve_read_namespaces("team-z", ctx)


def test_read_resolution_no_header_defaults_to_default_only():
    assert mcp_server._resolve_read_namespaces(None, None) == ["default"]


# ---- _resolve_write_namespace: write=home default, narrow, reject ---------


def test_write_resolution_omitted_uses_home_entry():
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    assert mcp_server._resolve_write_namespace(None, ctx) == "team-a"


def test_write_resolution_picks_named_entry_within_set():
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    assert mcp_server._resolve_write_namespace("team-b", ctx) == "team-b"


def test_write_resolution_rejects_namespace_outside_allowed_set():
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    with pytest.raises(ValueError, match="outside the allowed set"):
        mcp_server._resolve_write_namespace("team-z", ctx)


def test_write_resolution_no_header_defaults_to_default():
    assert mcp_server._resolve_write_namespace(None, None) == "default"


# ---- tool-level wiring: search sends the resolved namespaces filter -------


def test_search_memory_sends_whole_allowed_set_when_omitted(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    asyncio.run(mcp_server.search_memory(query="q", ctx=ctx))
    assert captured["json"]["namespaces"] == ["team-a", "team-b"]


def test_search_memory_narrows_to_requested_namespace(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    asyncio.run(mcp_server.search_memory(query="q", namespace="team-b", ctx=ctx))
    assert captured["json"]["namespaces"] == ["team-b"]


def test_search_memory_rejects_namespace_outside_allowed_set(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the backend")

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a"})
    with pytest.raises(ValueError, match="outside the allowed set"):
        asyncio.run(mcp_server.search_memory(query="q", namespace="team-z", ctx=ctx))


def test_search_all_and_search_code_send_whole_allowed_set(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    asyncio.run(mcp_server.search_all(query="q", ctx=ctx))
    assert captured["json"]["namespaces"] == ["team-a", "team-b"]
    asyncio.run(mcp_server.search_code(query="q", ctx=ctx))
    assert captured["json"]["namespaces"] == ["team-a", "team-b"]


def test_deep_search_sends_whole_allowed_set(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200, json={"evidence": [], "trace": [], "hops_used": 0, "stopped_reason": "done"}
        )

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    asyncio.run(mcp_server.deep_search(query="q", ctx=ctx))
    assert captured["json"]["namespaces"] == ["team-a", "team-b"]


def test_deep_search_rejects_namespace_outside_allowed_set(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the backend")

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a"})
    with pytest.raises(ValueError, match="outside the allowed set"):
        asyncio.run(mcp_server.deep_search(query="q", namespace="team-z", ctx=ctx))


# ---- tool-level wiring: write tools land in the home entry -----------------


def test_save_memory_uses_home_entry_when_omitted(monkeypatch):
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
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    asyncio.run(mcp_server.save_memory("distilled content", ctx=ctx))
    assert captured["json"]["namespace"] == "team-a"


def test_save_memory_picks_named_entry_within_set(monkeypatch):
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
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    asyncio.run(mcp_server.save_memory("distilled content", namespace="team-b", ctx=ctx))
    assert captured["json"]["namespace"] == "team-b"


def test_save_memory_rejects_namespace_outside_allowed_set(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the backend")

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a"})
    with pytest.raises(ValueError, match="outside the allowed set"):
        asyncio.run(mcp_server.save_memory("distilled content", namespace="team-z", ctx=ctx))


def test_ingest_document_uses_home_entry_when_omitted(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            202, json={"job_id": "job-1", "status": "queued", "status_url": "/ingest/jobs/job-1"}
        )

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a, team-b"})
    asyncio.run(mcp_server.ingest_document("# Guide", "guide.md", ctx=ctx))
    assert b'name="namespace"' in captured["body"]
    assert b"team-a" in captured["body"]


def test_ingest_document_rejects_namespace_outside_allowed_set(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the backend")

    _patch_client(monkeypatch, handler)
    ctx = FakeCtx(headers={"X-Memory-Namespaces": "team-a"})
    with pytest.raises(ValueError, match="outside the allowed set"):
        asyncio.run(mcp_server.ingest_document("# Guide", "guide.md", namespace="team-z", ctx=ctx))
