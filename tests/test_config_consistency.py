"""Unit tests for explicit env failures and the reranker's configured timeout.
No DB, no vLLM.
"""

from __future__ import annotations

import asyncio

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
    ],
)
def test_missing_env_var_raises_with_name(monkeypatch, name, action):
    monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match=name):
        action()


def test_rerank_uses_service_timeout(monkeypatch):
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
    assert captured.get("timeout") == config.SERVICE_TIMEOUT_SECONDS
