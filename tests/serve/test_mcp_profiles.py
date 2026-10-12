"""The three MCP profile tools: what they send, which key they forward, and their errors.

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
                "query": dict(request.url.params),
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


def test_exactly_three_profile_tools_are_registered():
    tools = _tools()
    assert {name for name in tools if "profile" in name} == {
        "get_my_profile",
        "update_my_profile",
        "update_user_profile",
    }
    assert set(tools["get_my_profile"].inputSchema["required"]) == {"owner"}
    assert set(tools["update_my_profile"].inputSchema["required"]) == {"owner", "content"}
    assert set(tools["update_user_profile"].inputSchema["required"]) == {
        "owner",
        "content",
        "reason",
        "base_version",
    }


def test_update_user_profile_description_names_the_consent_step():
    description = _tools()["update_user_profile"].description
    assert "confirmed=true" in description
    assert "needs_consent" in description


def test_get_my_profile_reads_the_profile_with_the_request_key(monkeypatch):
    body = {"owner": "claude-code", "self_version": 2, "user_version": 1, "user": None}
    captured = _capture(monkeypatch, 200, body)
    ctx = FakeCtx({"X-API-Key": "agents-key"})
    result = asyncio.run(tools.get_my_profile("claude-code", ctx=ctx))
    assert result == body
    assert captured == [
        {
            "method": "GET",
            "path": "/profiles",
            "query": {"owner": "claude-code"},
            "key": "agents-key",
            "json": None,
        }
    ]


def test_update_my_profile_puts_the_self_part_with_the_request_key(monkeypatch):
    captured = _capture(monkeypatch, 200, {"status": "written", "version": 3})
    ctx = FakeCtx({"X-API-Key": "agents-key"})
    result = asyncio.run(tools.update_my_profile("claude-code", "Rules.", ctx=ctx))
    assert result == {"status": "written", "version": 3}
    assert captured == [
        {
            "method": "PUT",
            "path": "/profiles/self",
            "query": {},
            "key": "agents-key",
            "json": {"owner": "claude-code", "content": "Rules."},
        }
    ]


def _backend(monkeypatch, dry_run, write, calls=None):
    """A fake REST backend answering a dry run and a real write of PUT /profiles/user."""
    calls = [] if calls is None else calls

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        status, payload = dry_run if body.get("dry_run") else write
        return httpx.Response(status, json=payload)

    monkeypatch.setattr(
        rest_client,
        "client",
        lambda: httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        ),
    )
    return calls


VALID = (200, {"status": "valid", "version": 1, "current": "Lives in Busan."})
WRITTEN = (200, {"status": "written", "version": 2})
CTX = FakeCtx({"X-API-Key": "agents-key"})


def _update(**fields):
    arguments = {
        "owner": "natsume",
        "content": "Lives in Seoul.",
        "reason": "they moved",
        "base_version": 1,
        **fields,
    }
    return asyncio.run(tools.update_user_profile(ctx=CTX, **arguments))


def test_update_user_profile_validates_then_writes_once_confirmed(monkeypatch):
    calls = _backend(monkeypatch, VALID, WRITTEN)
    assert _update(confirmed=True) == {"status": "written", "version": 2}
    common = {
        "owner": "natsume",
        "content": "Lives in Seoul.",
        "reason": "they moved",
        "base_version": 1,
    }
    assert calls == [
        {**common, "consent_method": "conversation", "dry_run": True},
        {**common, "consent_method": "conversation"},
    ]


def test_update_user_profile_without_confirmation_returns_the_prompt_and_writes_nothing(
    monkeypatch,
):
    calls = _backend(monkeypatch, VALID, WRITTEN)
    result = _update()
    assert result["status"] == "needs_consent"
    assert result["summary"].startswith(
        "memory-base asks for your consent to change how natsume knows you "
        "(user profile v1 -> v2):\nReason: they moved\n"
    )
    assert len(calls) == 1 and calls[0]["dry_run"] is True


def test_unchanged_content_asks_no_consent(monkeypatch):
    unchanged = (200, {"status": "unchanged", "version": 1})
    calls = _backend(monkeypatch, unchanged, WRITTEN)
    assert _update() == {"status": "unchanged", "version": 1}
    assert len(calls) == 1


def test_a_stale_base_is_refused_before_any_consent(monkeypatch):
    calls = _backend(monkeypatch, (409, {"error": "stale", "version": 5}), WRITTEN)
    with pytest.raises(ValueError) as exc:
        _update(confirmed=True)
    message = str(exc.value)
    assert "stale" in message
    assert "version 5" in message
    assert "get_my_profile" in message
    assert "not enough" in message
    assert len(calls) == 1


def test_a_base_that_advances_after_the_dry_run_is_stale_on_the_write(monkeypatch):
    _backend(monkeypatch, VALID, (409, {"error": "stale", "version": 2}))
    with pytest.raises(ValueError, match="stale: the user profile is now at version 2"):
        _update(confirmed=True)


LONG = "\n".join(f"line {i}" for i in range(40))


@pytest.mark.parametrize(
    ("current", "content", "base", "expected"),
    [
        (LONG, LONG.replace("line 4", "line four"), 3, "--- v3\n+++ v4\n"),
        ("", "Lives in Seoul.", 0, "Full replacement:\nLives in Seoul."),
        ("alpha\nbeta", "gamma", 1, "Full replacement:\ngamma"),
        ("Lives in Busan.", "", 2, "Clear the user profile. Current text:\nLives in Busan."),
    ],
    ids=["diff", "new profile", "diff longer than content", "clear"],
)
def test_change_body_picks_the_shorter_honest_form(current, content, base, expected):
    body = tools.change_body(current, content, base)
    assert body.startswith(expected)
    if expected.startswith("---"):
        assert "-line 4\n+line four" in body


def test_other_profile_errors_surface_the_backend_reason(monkeypatch):
    _capture(monkeypatch, 403, {"error": "owner 'natsume' is not permitted for this key"})
    with pytest.raises(ValueError, match="^owner 'natsume' is not permitted for this key$"):
        asyncio.run(tools.update_my_profile("natsume", "x"))
    _capture(monkeypatch, 409, {"error": "stale"})
    with pytest.raises(ValueError, match="^stale$"):
        asyncio.run(tools.update_my_profile("natsume", "x"))
    _capture(monkeypatch, 403, {"error": "owner 'natsume' is not permitted for this key"})
    with pytest.raises(ValueError, match="^owner 'natsume' is not permitted for this key$"):
        asyncio.run(tools.get_my_profile("natsume"))
