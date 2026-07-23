"""Shared fixtures for the test suite."""

from __future__ import annotations

import httpx
import pytest


@pytest.fixture()
def rest_in_process(monkeypatch):
    """Route the MCP proxy at the REST app in-process (no server on :8010).

    Integration tests that invoke MCP tools exercise the full stack —
    proxy -> Starlette app -> notes/search -> real DB/embedder — without
    binding a port.
    """
    from memory_base.serve import api, mcp_server

    def _client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
        )

    monkeypatch.setattr(mcp_server, "_client", _client)
