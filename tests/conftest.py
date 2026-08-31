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


@pytest.fixture(autouse=True)
def table_query_password_for_unit_tests(request, monkeypatch):
    """Unit schema fakes receive a password without requiring deployment configuration."""
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setenv(
            "TABLES_QUERY_PASSWORD", os.getenv("TABLES_QUERY_PASSWORD") or "test-only"
        )


@pytest.fixture(autouse=True)
def isolated_access_log_buffer():
    """Rows buffered by one test's /search calls must never be flushed into the
    real retrieval_log by a later integration test in the same run."""
    from memory_base.serve import access_log

    access_log._pending_logs.clear()
    access_log._pending_hits.clear()
    yield
    access_log._pending_logs.clear()
    access_log._pending_hits.clear()


@pytest.fixture(autouse=True)
def isolate_unit_app_lifespan(request, monkeypatch):
    """Keep ordinary tests independent of Postgres-backed worker startup."""
    if request.node.get_closest_marker("integration") is not None:
        return

    from memory_base.serve import access_log, job_store

    async def initialize():
        return None

    def start_workers():
        return []

    async def stop_workers(tasks):
        assert tasks == []

    def start_flusher():
        return None

    async def stop_flusher(task):
        assert task is None

    monkeypatch.setattr(job_store, "initialize", initialize)
    monkeypatch.setattr(job_store, "start_workers", start_workers)
    monkeypatch.setattr(job_store, "stop_workers", stop_workers)
    monkeypatch.setattr(access_log, "start_flusher", start_flusher)
    monkeypatch.setattr(access_log, "stop_flusher", stop_flusher)


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
