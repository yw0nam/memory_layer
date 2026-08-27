"""POST /search when the retrieval backends are down, and the query length bound.

Fails today: an unreachable embedder/reranker escapes search_route as a 500
traceback, and a multi-kilobyte query reaches search() and the access log
unchanged.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.retrieval.search import UpstreamUnavailable
from memory_base.serve import api

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


@pytest.fixture(autouse=True)
def _empty_access_log_buffer(monkeypatch):
    monkeypatch.setattr(api.access_log, "_pending_logs", [])
    monkeypatch.setattr(api.access_log, "_pending_hits", {})


def _raising(service: str):
    async def fake_search(query, **kwargs):
        raise UpstreamUnavailable(service)

    return fake_search


@pytest.mark.parametrize("service", ["embedding", "reranking"])
def test_unreachable_backend_returns_503_naming_the_service(monkeypatch, service):
    monkeypatch.setattr(api, "search", _raising(service))
    response = client.post("/search", json={"query": "what did we decide"})
    assert response.status_code == 503
    message = response.json()["error"]
    assert service in message
    assert "memory cannot be attached right now" in message


def test_unreachable_backend_records_no_access_log_row(monkeypatch):
    monkeypatch.setattr(api, "search", _raising("embedding"))
    client.post("/search", json={"query": "what did we decide"})
    assert api.access_log._pending_logs == []


def test_long_query_is_truncated_before_search(monkeypatch):
    captured = {}

    async def fake_search(query, **kwargs):
        captured["query"] = query
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "x" * (api.MAX_QUERY_CHARS + 500)})
    assert response.status_code == 200
    assert captured["query"] == "x" * api.MAX_QUERY_CHARS


def test_long_query_is_truncated_in_the_access_log(monkeypatch):
    async def fake_search(query, **kwargs):
        return []

    monkeypatch.setattr(api, "search", fake_search)
    client.post("/search", json={"query": "y" * (api.MAX_QUERY_CHARS + 500)})
    logged_query, _, _, _ = api.access_log._pending_logs[0]
    assert logged_query == "y" * api.MAX_QUERY_CHARS


def test_query_within_the_bound_is_untouched(monkeypatch):
    captured = {}

    async def fake_search(query, **kwargs):
        captured["query"] = query
        return []

    monkeypatch.setattr(api, "search", fake_search)
    client.post("/search", json={"query": "short question"})
    assert captured["query"] == "short question"
