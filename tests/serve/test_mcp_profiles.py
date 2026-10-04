"""The two MCP profile tools: what they send, which key they forward, and their errors.

No DB/network: rest_client.client is mocked via httpx.MockTransport.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from memory_base.serve import mcp_server
from memory_base.serve.common import rest_client
from memory_base.serve.profiles import tools


class FakeHeaders:
    def __init__(self, headers):
        self._headers = {k.lower(): v for k, v in headers.items()}

    def get(self, key, default=None):
        return self._headers.get(key.lower(), default)


class FakeCtx:
    def __init__(self, headers):
        request = type("Request", (), {"headers": FakeHeaders(headers)})()
        self.request_context = type("RequestContext", (), {"request": request})()


def _capture(monkeypatch, status, body):
    captured = []

    def handler(request):
        captured.append(
            {
                "method": request.method,
                "path": request.url.path,
                "key": request.headers.get("x-api-key"),
                "json": json.loads(request.content) if request.content else None,
            }
        )
        return httpx.Response(status, json=body)

    def fake_client():
        return httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(rest_client, "client", fake_client)
    return captured


def _tools():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.list_tools()
            return {t.name: t for t in result.tools}

    return asyncio.run(_run())


def test_exactly_two_profile_tools_are_registered():
    tools = _tools()
    assert {name for name in tools if "profile" in name} == {
        "update_my_profile",
        "propose_user_profile",
    }
    assert not {name for name in tools if "approve" in name or "reject" in name}
    assert set(tools["update_my_profile"].inputSchema["required"]) == {"owner", "content"}
    assert set(tools["propose_user_profile"].inputSchema["required"]) == {
        "owner",
        "content",
        "reason",
        "base_version",
    }


def test_the_tool_descriptions_state_who_writes_what():
    tools = _tools()
    own = tools["update_my_profile"].description
    assert "replaces the whole" in own
    assert "session start" in own
    assert "never put facts about the user" in own.lower()
    propose = tools["propose_user_profile"].description
    assert "approves" in propose
    assert "profile-approval skill" in propose
    assert "never approve" in propose.lower()
    assert "approval command" in propose


def test_update_my_profile_puts_the_self_part_with_the_request_key(monkeypatch):
    captured = _capture(monkeypatch, 200, {"status": "written", "version": 3})
    ctx = FakeCtx({"X-API-Key": "agents-key"})
    result = asyncio.run(tools.update_my_profile("claude-code", "Rules.", ctx=ctx))
    assert result == {"status": "written", "version": 3}
    assert captured == [
        {
            "method": "PUT",
            "path": "/profiles/self",
            "key": "agents-key",
            "json": {"owner": "claude-code", "content": "Rules."},
        }
    ]


def test_propose_user_profile_posts_a_proposal_with_the_stdio_key(monkeypatch):
    monkeypatch.setenv("MEMORY_API_KEY", "stdio-key")
    captured = _capture(monkeypatch, 201, {"id": 4, "status": "pending", "superseded": 2})
    result = asyncio.run(tools.propose_user_profile("natsume", "Lives in Seoul.", "they moved", 1))
    assert result == {"id": 4, "status": "pending", "superseded": 2}
    assert captured == [
        {
            "method": "POST",
            "path": "/profiles/user/proposals",
            "key": "stdio-key",
            "json": {
                "owner": "natsume",
                "content": "Lives in Seoul.",
                "reason": "they moved",
                "base_version": 1,
            },
        }
    ]


def test_a_stale_proposal_keeps_the_current_version_and_says_to_refresh(monkeypatch):
    _capture(monkeypatch, 409, {"error": "stale", "version": 5})
    with pytest.raises(ValueError) as exc:
        asyncio.run(tools.propose_user_profile("claude-code", "x", "r", 3))
    message = str(exc.value)
    assert "stale" in message
    assert "version 5" in message
    assert "refresh" in message.lower()
    assert "session-start" in message
    assert "base_version" in message
    assert "not enough" in message


def test_other_profile_errors_surface_the_backend_reason(monkeypatch):
    _capture(monkeypatch, 403, {"error": "owner 'natsume' is not permitted for this key"})
    with pytest.raises(ValueError, match="^owner 'natsume' is not permitted for this key$"):
        asyncio.run(tools.update_my_profile("natsume", "x"))
    _capture(monkeypatch, 409, {"error": "stale"})
    with pytest.raises(ValueError, match="^stale$"):
        asyncio.run(tools.update_my_profile("natsume", "x"))
