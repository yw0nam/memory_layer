"""Unit tests for the message-lane MCP tools: thin proxies over the REST API.

Same harness as tests/serve/test_mcp_proxy.py: monkeypatch
``mcp_server._client`` with a MockTransport and assert the exact REST call each
tool issues. No DB, no network.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from memory_base.serve import mcp_server

MESSAGE_ID = "5f0d9d44-9a9d-4f0e-b7f6-6fa1e2b3c4d5"


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=mcp_server.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(mcp_server, "_client", fake_client)


def _tools():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.list_tools()
            return {t.name: t for t in result.tools}

    return asyncio.run(_run())


def _call(tool_name, monkeypatch, handler, **arguments):
    calls = []

    def handler_wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    _patch_client(monkeypatch, handler_wrapped)

    async def _run():
        from mcp.shared.memory import create_connected_server_and_client_session

        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.call_tool(tool_name, arguments)
            return result.structuredContent["result"]

    return asyncio.run(_run()), calls


# ---- registration -------------------------------------------------------------


def test_message_tools_are_registered():
    tools = _tools()
    assert {"send_message", "list_messages", "claim_message", "cancel_message"} <= set(tools)


def test_send_message_schema_requires_subject_result_author():
    tools = _tools()
    required = set(tools["send_message"].inputSchema["required"])
    assert {"subject", "result", "author"} <= required


# ---- send_message ----------------------------------------------------------------


def test_send_message_posts_report_fields(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={
                "id": MESSAGE_ID,
                "namespace": "default",
                "purpose": "message",
                "scope": None,
                "subject": "s",
                "status": "info",
                "delivery": "pending",
                "author": "claude-code",
                "created_at": "2026-02-03T10:00:00+00:00",
                "expires_at": "2026-02-10T10:00:00+00:00",
                "content": "# S",
            },
        )

    payload, calls = _call(
        "send_message",
        monkeypatch,
        handler,
        subject="s",
        result="r",
        author="claude-code",
        status="info",
        next_step=None,
        verification=None,
        refs=None,
        scope=None,
        namespace=None,
        idempotency_key=None,
        expires_at=None,
    )
    assert len(calls) == 1
    request = calls[0]
    assert request.method == "POST"
    assert request.url.path == "/messages"
    body = json.loads(request.content)
    assert body["subject"] == "s"
    assert body["result"] == "r"
    assert body["author"] == "claude-code"
    assert "next" not in body
    assert payload["id"] == MESSAGE_ID


def test_send_message_maps_next_step_to_next_and_sends_scope(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": MESSAGE_ID})

    _call(
        "send_message",
        monkeypatch,
        handler,
        subject="handoff state",
        result="r",
        author="claude-code",
        status="in_progress",
        next_step="resume the replay",
        scope="repo:github.com/o/r",
    )
    body = captured["body"]
    assert body["next"] == "resume the replay"
    assert body["scope"] == "repo:github.com/o/r"
    assert body["status"] == "in_progress"
    assert "next_step" not in body


def test_send_message_surfaces_409_conflict_as_tool_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "idempotency_key already used"})

    with pytest.raises(ValueError, match="idempotency_key already used"):
        _call(
            "send_message",
            monkeypatch,
            handler,
            subject="s",
            result="r",
            author="claude-code",
            idempotency_key="run-1",
        )


# ---- list_messages -----------------------------------------------------------------


def test_list_messages_gets_with_filters(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    payload, calls = _call(
        "list_messages",
        monkeypatch,
        handler,
        namespace="team-a",
        purpose="handoff",
        scope="repo:github.com/o/r",
        subject="fix login",
        limit=10,
    )
    assert calls[0].method == "GET"
    assert calls[0].url.path == "/messages"
    assert captured["params"] == {
        "namespace": "team-a",
        "purpose": "handoff",
        "scope": "repo:github.com/o/r",
        "subject": "fix login",
        "limit": "10",
    }
    assert payload == []


def test_list_messages_omits_unset_filters(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    _call("list_messages", monkeypatch, handler)
    assert captured["params"] == {}


# ---- claim / cancel ------------------------------------------------------------------


def test_claim_message_posts_to_claim_route(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": MESSAGE_ID, "status": "info", "delivery": "claimed"})

    payload, calls = _call("claim_message", monkeypatch, handler, message_id=MESSAGE_ID)
    assert calls[0].method == "POST"
    assert calls[0].url.path == f"/messages/{MESSAGE_ID}/claim"
    assert payload["delivery"] == "claimed"
    assert payload["status"] == "info"


def test_cancel_message_deletes_by_id(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"id": MESSAGE_ID, "status": "info", "delivery": "cancelled"}
        )

    payload, calls = _call("cancel_message", monkeypatch, handler, message_id=MESSAGE_ID)
    assert calls[0].method == "DELETE"
    assert calls[0].url.path == f"/messages/{MESSAGE_ID}"
    assert payload["delivery"] == "cancelled"
    assert payload["status"] == "info"


def test_claim_message_surfaces_404_as_tool_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no claimable message"})

    with pytest.raises(ValueError, match="no claimable message"):
        _call("claim_message", monkeypatch, handler, message_id=MESSAGE_ID)
