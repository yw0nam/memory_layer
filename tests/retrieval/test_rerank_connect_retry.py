"""The reranker call: a 2 s connect timeout, one retry on a connect failure, none otherwise.

Fails today: _rerank uses one 10 s timeout for connect and read and never retries.
"""

from __future__ import annotations

import asyncio
import json

import httpcore
import pytest

from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import UpstreamUnavailable
from memory_base.retrieval.search import _rerank


def _http_response(status: int, body: dict) -> bytes:
    payload = json.dumps(body).encode()
    head = f"HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {len(payload)}\r\n\r\n"
    return head.encode() + payload


class _FlakyBackend(httpcore.AsyncMockBackend):
    """Fails the first `failures` connects with ConnectTimeout, then serves the response."""

    def __init__(self, buffer, failures):
        super().__init__(buffer)
        self.failures = failures
        self.timeouts: list = []

    async def connect_tcp(self, host, port, timeout=None, **kwargs):
        self.timeouts.append(timeout)
        if len(self.timeouts) <= self.failures:
            raise httpcore.ConnectTimeout("stalled")
        return await super().connect_tcp(host, port, timeout, **kwargs)


def _backend(monkeypatch, failures, status=200):
    body = {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]}
    backend = _FlakyBackend([_http_response(status, body)], failures)
    monkeypatch.setattr("httpcore._async.connection_pool.AutoBackend", lambda: backend)
    monkeypatch.setenv("RERANK_URL", "http://rerank.test")
    return backend


def _hits():
    return [Hit(source="memory", ref=f"note:{i}", text="body", ts=0.0, rrf=0.02) for i in (1, 2)]


def test_connect_stall_is_retried_once_and_reorders_hits(monkeypatch):
    backend = _backend(monkeypatch, failures=1)
    hits = asyncio.run(_rerank("q", _hits()))
    assert [h.ref for h in hits] == ["note:2", "note:1"]
    assert len(backend.timeouts) == 2


def test_connect_stall_twice_is_upstream_unavailable_after_two_attempts(monkeypatch):
    backend = _backend(monkeypatch, failures=2)
    with pytest.raises(UpstreamUnavailable) as excinfo:
        asyncio.run(_rerank("q", _hits()))
    assert excinfo.value.service == "reranking"
    assert len(backend.timeouts) == 2


def test_connect_uses_the_short_connect_timeout(monkeypatch):
    backend = _backend(monkeypatch, failures=0)
    asyncio.run(_rerank("q", _hits()))
    assert backend.timeouts == [2]


def test_http_error_status_is_not_retried(monkeypatch):
    backend = _backend(monkeypatch, failures=0, status=500)
    with pytest.raises(UpstreamUnavailable) as excinfo:
        asyncio.run(_rerank("q", _hits()))
    assert excinfo.value.service == "reranking"
    assert len(backend.timeouts) == 1
