"""Unit tests for the multi-repo MCP tools (thin REST proxies)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from memory_base.serve import mcp_server


def _mock_client(handler):
    def factory():
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://testserver"
        )

    return factory


def test_ingest_repo_returns_job_id_and_status_url(monkeypatch):
    def handler(request):
        assert request.url.path == "/repos"
        assert request.method == "POST"
        return httpx.Response(
            202,
            json={
                "job_id": "j1",
                "name": "repo",
                "status": "queued",
                "status_url": "/repos/jobs/j1",
            },
        )

    monkeypatch.setattr(mcp_server, "_client", _mock_client(handler))
    result = asyncio.run(mcp_server.ingest_repo("https://github.com/o/repo.git"))
    assert result == {"job_id": "j1", "status_url": "/repos/jobs/j1"}


def test_remove_repo_returns_job_id_and_status_url(monkeypatch):
    def handler(request):
        assert request.method == "DELETE"
        assert request.url.path == "/repos/repo"
        return httpx.Response(
            202,
            json={
                "job_id": "j2",
                "name": "repo",
                "status": "queued",
                "status_url": "/repos/jobs/j2",
            },
        )

    monkeypatch.setattr(mcp_server, "_client", _mock_client(handler))
    result = asyncio.run(mcp_server.remove_repo("repo"))
    assert result == {"job_id": "j2", "status_url": "/repos/jobs/j2"}


def test_list_repos_returns_backend_list(monkeypatch):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/repos"
        return httpx.Response(
            200,
            json=[{"name": "repo", "url": "u", "branch": "main", "head": "abc1234", "chunks": 3}],
        )

    monkeypatch.setattr(mcp_server, "_client", _mock_client(handler))
    result = asyncio.run(mcp_server.list_repos())
    assert result[0]["name"] == "repo"


def test_ingest_repo_raises_backend_error(monkeypatch):
    def handler(request):
        return httpx.Response(400, json={"error": "invalid url"})

    monkeypatch.setattr(mcp_server, "_client", _mock_client(handler))
    with pytest.raises(ValueError, match="invalid url"):
        asyncio.run(mcp_server.ingest_repo("ftp://bad"))
