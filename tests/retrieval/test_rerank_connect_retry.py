"""The reranker call: a 2 s connect timeout, one retry on a connect failure, none otherwise,
and a malformed upstream body reported as the reranker being unavailable.
"""

from __future__ import annotations

import asyncio
import json

import httpcore
import pytest

from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import UpstreamUnavailable
from memory_base.retrieval.search import _rerank
from memory_base.retrieval.search import rerank_scores


def _http_response(status: int, body: dict | bytes) -> bytes:
    payload = body if isinstance(body, bytes) else json.dumps(body).encode()
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


def _backend(monkeypatch, failures, status=200, body=None):
    if body is None:
        body = {
            "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
        }
    backend = _FlakyBackend([_http_response(status, body)], failures)
    monkeypatch.setattr("httpcore._async.connection_pool.AutoBackend", lambda: backend)
    monkeypatch.setenv("RERANK_URL", "http://rerank.test")
    return backend


def _hits():
    return [Hit(ref=f"note:{i}", text="body", ts=0.0, rrf=0.02) for i in (1, 2)]


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


def test_rerank_scores_are_in_input_order_whatever_order_the_upstream_returns(monkeypatch):
    _backend(monkeypatch, failures=0)
    assert asyncio.run(rerank_scores("q", ["first", "second"])) == [0.1, 0.9]


def _results(*items):
    return {"results": [{"index": i, "relevance_score": score} for i, score in items]}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param({}, id="results-missing"),
        pytest.param({"results": {"0": 1}}, id="results-not-a-list"),
        pytest.param(_results((0, 0.5)), id="index-missing"),
        pytest.param(_results((0, 0.5), (0, 0.6), (1, 0.7)), id="index-duplicate"),
        pytest.param(_results((0, 0.5), (2, 0.6)), id="index-out-of-range"),
        pytest.param({"results": [{"index": "1", "relevance_score": 0.5}]}, id="index-not-int"),
        pytest.param({"results": [{"index": 0}]}, id="score-absent"),
        pytest.param(b'{"results": [{"index": 0, "relevance_score": NaN}]}', id="score-nan"),
    ],
)
def test_malformed_upstream_body_is_upstream_unavailable(monkeypatch, body):
    _backend(monkeypatch, failures=0, body=body)
    with pytest.raises(UpstreamUnavailable) as excinfo:
        asyncio.run(rerank_scores("q", ["a", "b"]))
    assert excinfo.value.service == "reranking"


def test_no_request_is_made_for_zero_hits(monkeypatch):
    backend = _backend(monkeypatch, failures=0)
    assert asyncio.run(_rerank("q", [])) == []
    assert backend.timeouts == []
