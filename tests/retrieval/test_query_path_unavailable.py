"""The query path's vLLM calls: bounded by the query timeout, unreachable -> UpstreamUnavailable.

Fails today: memory_base.retrieval.search has no UpstreamUnavailable and no
_embed_query, and _rerank still uses the 120s ingest timeout and lets the raw
httpx/openai exception escape to the caller.
"""

from __future__ import annotations

import asyncio

import httpx
import openai
import pytest

from memory_base.core.config import QUERY_TIMEOUT_SECONDS
from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import UpstreamUnavailable
from memory_base.retrieval.search import _embed_query
from memory_base.retrieval.search import _rerank


@pytest.fixture(autouse=True)
def _endpoints(monkeypatch):
    monkeypatch.setenv("RERANK_URL", "http://rerank.test")
    monkeypatch.setenv("EMB_URL", "http://embed.test")


def _hit():
    return Hit(source="memory", ref="note:1", text="body", ts=0.0, rrf=0.02)


def _patch_httpx(monkeypatch, handler, captured: dict | None = None):
    real = httpx.AsyncClient

    def fake_client(*args, **kwargs):
        if captured is not None:
            captured.update(kwargs)
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)


# ---- rerank ---------------------------------------------------------------


def test_rerank_connect_failure_is_upstream_unavailable(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route to host", request=request)

    _patch_httpx(monkeypatch, handler)
    with pytest.raises(UpstreamUnavailable) as excinfo:
        asyncio.run(_rerank("q", [_hit()]))
    assert excinfo.value.service == "reranking"


def test_rerank_error_status_is_upstream_unavailable(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model not loaded")

    _patch_httpx(monkeypatch, handler)
    with pytest.raises(UpstreamUnavailable) as excinfo:
        asyncio.run(_rerank("q", [_hit()]))
    assert excinfo.value.service == "reranking"


def test_rerank_uses_the_query_path_timeout(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    _patch_httpx(monkeypatch, handler, captured)
    hits = asyncio.run(_rerank("q", [_hit()]))
    assert captured["timeout"] == QUERY_TIMEOUT_SECONDS
    assert hits[0].rerank_score == 0.9


# ---- query embedding ------------------------------------------------------


def test_embed_query_connection_error_is_upstream_unavailable(monkeypatch):
    async def boom(self, text, *, query=False):
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://embed.test"))

    monkeypatch.setattr("memory_base.core.config.VllmEmbedder.embed", boom)
    with pytest.raises(UpstreamUnavailable) as excinfo:
        asyncio.run(_embed_query("q"))
    assert excinfo.value.service == "embedding"


def test_embed_query_timeout_is_upstream_unavailable(monkeypatch):
    async def never(self, text, *, query=False):
        await asyncio.sleep(60)

    monkeypatch.setattr("memory_base.core.config.VllmEmbedder.embed", never)
    monkeypatch.setattr("memory_base.retrieval.search.QUERY_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(UpstreamUnavailable) as excinfo:
        asyncio.run(_embed_query("q"))
    assert excinfo.value.service == "embedding"


def test_embed_query_returns_a_vector_literal(monkeypatch):
    async def fake(self, text, *, query=False):
        import numpy as np

        return np.asarray([0.5, -0.25], dtype=np.float16)

    monkeypatch.setattr("memory_base.core.config.VllmEmbedder.embed", fake)
    assert asyncio.run(_embed_query("q")) == "[0.500000,-0.250000]"
