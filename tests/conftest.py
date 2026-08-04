"""Shared fixtures for the test suite."""

from __future__ import annotations

import os

import httpx
import pytest


@pytest.fixture(autouse=True)
def model_names(monkeypatch):
    """Tests that fake the backends still need the names to resolve; real ones win."""
    for name in ("LLM_MODEL", "EMB_MODEL", "RERANK_MODEL"):
        monkeypatch.setenv(name, os.getenv(name) or "test-model")


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
    # mcp_server has no request context here (ctx=None), so it falls back to this env var;
    # tests/serve/conftest.py's auth stub resolves "test-key" to an admin identity.
    monkeypatch.setenv("MEMORY_API_KEY", "test-key")
