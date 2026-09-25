"""A failing MCP tool call states why it failed: the backend's reason, a timeout, or no backend."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from memory_base.core.config import SERVICE_TIMEOUT_SECONDS
from memory_base.serve import mcp_server


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=mcp_server.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(mcp_server, "_client", fake_client)


def _list_repos_error(monkeypatch, handler) -> str:
    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError) as exc:
        asyncio.run(mcp_server.list_repos())
    return str(exc.value)


@pytest.mark.parametrize("status", [400, 404, 409, 500, 503])
def test_any_error_status_surfaces_the_backend_reason(monkeypatch, status):
    message = _list_repos_error(
        monkeypatch, lambda request: httpx.Response(status, json={"error": "embedder is down"})
    )
    assert message == "embedder is down"


@pytest.mark.parametrize("status", [307, 502])
def test_error_without_a_reason_payload_names_status_and_body(monkeypatch, status):
    message = _list_repos_error(
        monkeypatch, lambda request: httpx.Response(status, text="upstream proxy failure")
    )
    assert str(status) in message
    assert "upstream proxy failure" in message
    assert "GET /repos" in message


def test_timeout_names_the_backend_and_the_call(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("", request=request)

    message = _list_repos_error(monkeypatch, handler)
    assert "timed out" in message
    assert mcp_server.REST_URL in message
    assert "GET /repos" in message


def test_unreachable_backend_names_the_backend_and_the_call(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("All connection attempts failed", request=request)

    message = _list_repos_error(monkeypatch, handler)
    assert "unreachable" in message
    assert mcp_server.REST_URL in message
    assert "GET /repos" in message
    assert "All connection attempts failed" in message


def test_proxy_waits_out_the_backends_own_ceilings():
    assert mcp_server._client().timeout.read >= SERVICE_TIMEOUT_SECONDS
