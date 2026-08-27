"""Unit tests for explicit env failures and the reranker's configured timeout.
No DB, no vLLM.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from memory_base.core import config
from memory_base.retrieval import search as search_mod


@pytest.mark.parametrize(
    "name, action",
    [
        ("LLM_URL", lambda: config.llm_client()),
        ("EMB_URL", lambda: config.VllmEmbedder()),
        (
            "RERANK_URL",
            lambda: asyncio.run(
                search_mod._rerank(
                    "q", [search_mod.Hit(source="memory", ref="r", text="t", ts=0.0)]
                )
            ),
        ),
        ("DB_URL", lambda: config.db_url()),
        ("LLM_MODEL", lambda: config.llm_model()),
        ("EMB_MODEL", lambda: config.emb_model()),
        ("RERANK_MODEL", lambda: config.rerank_model()),
    ],
)
def test_missing_env_var_raises_with_name(monkeypatch, name, action):
    monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match=name):
        action()


def test_oversample_factor_is_gone():
    assert not hasattr(config, "OVERSAMPLE_FACTOR")


def test_rerank_uses_query_timeout(monkeypatch):
    monkeypatch.setenv("RERANK_URL", "http://fake")
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 1.0}]}

    class _FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json):
            del url, json
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    hit = search_mod.Hit(source="memory", ref="r", text="text", ts=0.0)
    asyncio.run(search_mod._rerank("q", [hit]))
    assert captured.get("timeout") == config.QUERY_TIMEOUT_SECONDS


def test_embed_many_orders_output_by_response_index_not_response_order(monkeypatch):
    monkeypatch.setenv("EMB_URL", "http://fake")
    embedder = config.VllmEmbedder()

    class _ShuffledEmbeddings:
        async def create(self, **kwargs):
            del kwargs
            # Response items arrive out of input order; each embedding encodes its true index.
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=2, embedding=[2.0]),
                    SimpleNamespace(index=0, embedding=[0.0]),
                    SimpleNamespace(index=1, embedding=[1.0]),
                ]
            )

    embedder._client = SimpleNamespace(embeddings=_ShuffledEmbeddings())
    vectors = asyncio.run(embedder.embed_many(["a", "b", "c"]))
    assert [float(v[0]) for v in vectors] == [0.0, 1.0, 2.0]
