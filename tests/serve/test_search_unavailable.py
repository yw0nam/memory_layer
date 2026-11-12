"""POST /search when the retrieval backends are down, and the query length bound.

An unreachable embedder/reranker answers 503, and a multi-kilobyte query is
truncated before it reaches search().
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.retrieval.search import UpstreamUnavailable
from memory_base.serve import api
from memory_base.serve.search import routes as search_routes

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})


def _raising(service: str):
    async def fake_search(query, **kwargs):
        raise UpstreamUnavailable(service)

    return fake_search


@pytest.mark.parametrize("service", ["embedding", "reranking"])
def test_unreachable_backend_returns_503_naming_the_service(monkeypatch, service):
    monkeypatch.setattr(search_routes, "search", _raising(service))
    response = client.post("/search", json={"query": "what did we decide"})
    assert response.status_code == 503
    message = response.json()["error"]
    assert service in message
    assert "memory cannot be attached right now" in message


def test_long_query_is_truncated_before_search(monkeypatch):
    captured = {}

    async def fake_search(query, **kwargs):
        captured["query"] = query
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "x" * (search_routes.MAX_QUERY_CHARS + 500)})
    assert response.status_code == 200
    assert captured["query"] == "x" * search_routes.MAX_QUERY_CHARS


def test_query_within_the_bound_is_untouched(monkeypatch):
    captured = {}

    async def fake_search(query, **kwargs):
        captured["query"] = query
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    client.post("/search", json={"query": "short question"})
    assert captured["query"] == "short question"


def test_unavailable_503_logs_service_cause_and_elapsed(monkeypatch, caplog):
    async def fake_search(query, **kwargs):
        raise UpstreamUnavailable("reranking") from TimeoutError("connect stalled")

    monkeypatch.setattr(search_routes, "search", fake_search)
    with caplog.at_level("WARNING"):
        response = client.post("/search", json={"query": "what did we decide"})
    assert response.status_code == 503
    assert response.json()["error"] == (
        "search unavailable: the reranking service is not responding, so memory cannot be attached right now"
    )
    [record] = [r for r in caplog.records if r.levelname == "WARNING"]
    line = record.getMessage()
    assert "reranking" in line and "TimeoutError" in line and "connect stalled" in line
    assert "elapsed" in line
