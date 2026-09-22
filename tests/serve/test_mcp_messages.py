"""Unit tests for the message-lane MCP tools: thin proxies over the REST API.

Same harness as tests/serve/test_mcp_proxy.py: monkeypatch
``mcp_server._client`` with a MockTransport, call the tool functions directly,
and assert the exact REST call each tool issues. Tool registration is checked
in-process via create_connected_server_and_client_session. No DB, no network.
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


def _capturing_handler(captured, status=201, body=None):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        captured["json"] = json.loads(request.content) if request.content else None
        return httpx.Response(status, json=body if body is not None else {"id": MESSAGE_ID})

    return handler


# ---- registration -------------------------------------------------------------


def test_message_tools_are_registered():
    tools = _tools()
    assert {"send_message", "list_messages", "claim_message", "cancel_message"} <= set(tools)


def test_send_message_schema_requires_subject_result_author():
    tools = _tools()
    required = set(tools["send_message"].inputSchema["required"])
    assert {"subject", "result", "author"} <= required


def test_claim_and_cancel_take_the_message_id():
    tools = _tools()
    assert "message_id" in tools["claim_message"].inputSchema["required"]
    assert "message_id" in tools["cancel_message"].inputSchema["required"]


def test_tool_docs_do_not_claim_a_public_delivery_field():
    tools = _tools()
    for name in ("send_message", "list_messages", "claim_message", "cancel_message"):
        assert "delivery" not in tools[name].description, name


def test_send_message_doc_does_not_claim_english_subjects():
    tools = _tools()
    assert "English" not in tools["send_message"].description


def test_list_messages_doc_lists_the_exact_public_fields():
    tools = _tools()
    description = tools["list_messages"].description
    for field in (
        "id",
        "namespace",
        "purpose",
        "scope",
        "subject",
        "status",
        "author",
        "created_at",
        "expires_at",
        "content",
    ):
        assert field in description, field


# ---- send_message ----------------------------------------------------------------


def test_send_message_posts_report_fields(monkeypatch):
    captured = {}
    row = {
        "id": MESSAGE_ID,
        "namespace": "default",
        "purpose": "message",
        "scope": None,
        "subject": "s",
        "status": "info",
        "author": "claude-code",
        "created_at": "2026-02-03T10:00:00+00:00",
        "expires_at": "2026-02-10T10:00:00+00:00",
        "content": "# S",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(201, json=row)

    _patch_client(monkeypatch, handler)
    payload = asyncio.run(
        mcp_server.send_message(
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
    )
    assert captured["method"] == "POST"
    assert captured["path"] == "/messages"
    assert captured["json"] == {
        "subject": "s",
        "result": "r",
        "author": "claude-code",
        "status": "info",
    }
    assert payload == row


def test_send_message_maps_next_step_to_next_and_sends_scope(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, _capturing_handler(captured))
    asyncio.run(
        mcp_server.send_message(
            subject="handoff state",
            result="r",
            author="claude-code",
            status="in_progress",
            next_step="resume the replay",
            scope="repo:github.com/o/r",
        )
    )
    body = captured["json"]
    assert body["next"] == "resume the replay"
    assert body["scope"] == "repo:github.com/o/r"
    assert body["status"] == "in_progress"
    assert "next_step" not in body


def test_send_message_forwards_optional_fields(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, _capturing_handler(captured))
    asyncio.run(
        mcp_server.send_message(
            subject="s",
            result="r",
            author="claude-code",
            verification={"command": "uv run pytest", "status": "passed", "result": "12 green"},
            refs=["https://example.com/a"],
            namespace="team-a",
            idempotency_key="run-1",
            expires_at="2026-03-01T00:00:00+00:00",
        )
    )
    body = captured["json"]
    assert body["verification"] == {
        "command": "uv run pytest",
        "status": "passed",
        "result": "12 green",
    }
    assert body["refs"] == ["https://example.com/a"]
    assert body["namespace"] == "team-a"
    assert body["idempotency_key"] == "run-1"
    assert body["expires_at"] == "2026-03-01T00:00:00+00:00"


def test_send_message_surfaces_409_conflict_as_tool_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "idempotency_key already used"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="idempotency_key already used"):
        asyncio.run(mcp_server.send_message(subject="s", result="r", author="claude-code"))


# ---- list_messages -----------------------------------------------------------------


def test_list_messages_gets_with_filters(monkeypatch):
    captured = {}
    _patch_client(monkeypatch, _capturing_handler(captured, status=200, body=[]))
    payload = asyncio.run(
        mcp_server.list_messages(
            namespace="team-a",
            purpose="handoff",
            scope="repo:github.com/o/r",
            subject="fix login",
            limit=10,
        )
    )
    assert captured["method"] == "GET"
    assert captured["path"] == "/messages"
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
    _patch_client(monkeypatch, _capturing_handler(captured, status=200, body=[]))
    asyncio.run(mcp_server.list_messages())
    assert captured["params"] == {}


# ---- claim / cancel ------------------------------------------------------------------


def test_claim_message_posts_to_claim_route(monkeypatch):
    captured = {}
    _patch_client(
        monkeypatch,
        _capturing_handler(captured, status=200, body={"id": MESSAGE_ID, "status": "info"}),
    )
    payload = asyncio.run(mcp_server.claim_message(message_id=MESSAGE_ID))
    assert captured["method"] == "POST"
    assert captured["path"] == f"/messages/{MESSAGE_ID}/claim"
    assert payload["status"] == "info"


def test_cancel_message_deletes_by_id(monkeypatch):
    captured = {}
    _patch_client(
        monkeypatch,
        _capturing_handler(
            captured,
            status=200,
            body={"id": MESSAGE_ID, "status": "info"},
        ),
    )
    payload = asyncio.run(mcp_server.cancel_message(message_id=MESSAGE_ID))
    assert captured["method"] == "DELETE"
    assert captured["path"] == f"/messages/{MESSAGE_ID}"
    assert payload["status"] == "info"


def test_claim_and_cancel_reject_a_non_uuid_message_id(monkeypatch):
    """A traversal id must never be interpolated into the REST path."""

    def handler(request):
        raise AssertionError(f"no request expected, got {request.url}")

    _patch_client(monkeypatch, handler)
    for bad in ("../namespaces/default", "not-a-uuid", "5f0d9d44-9a9d-4f0e-b7f6-6fa1e2b3c4d5/x"):
        with pytest.raises(ValueError, match="message_id"):
            asyncio.run(mcp_server.claim_message(message_id=bad))
        with pytest.raises(ValueError, match="message_id"):
            asyncio.run(mcp_server.cancel_message(message_id=bad))


def test_claim_message_surfaces_404_as_tool_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "no claimable message"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="no claimable message"):
        asyncio.run(mcp_server.claim_message(message_id=MESSAGE_ID))
