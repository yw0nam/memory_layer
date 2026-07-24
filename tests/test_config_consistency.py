"""Unit tests for explicit env failures, shared LLM client injection, and the
reranker's configured timeout. No DB, no vLLM.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from memory_base import common
from memory_base.adapters.base import Session
from memory_base.ingest import history
from memory_base.retrieval import search as search_mod


@pytest.mark.parametrize(
    "name, action",
    [
        ("LLM_URL", lambda: common.llm_client()),
        ("EMB_URL", lambda: common.VllmEmbedder()),
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


def test_distill_fanout_shares_one_llm_client():
    construction_count = 0

    class _FakeCompletions:
        async def create(self, **kwargs):
            del kwargs
            content = json.dumps(
                {
                    "one_line_question": "q",
                    "summary": "s",
                    "resolution": "r",
                    "references": [],
                    "decisions": [],
                }
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class _FakeLLM:
        def __init__(self):
            nonlocal construction_count
            construction_count += 1
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    sessions = [
        Session(session_id=f"s{i}", messages=[], transcript="t", ts_last_active=0.0)
        for i in range(3)
    ]
    semaphore = asyncio.Semaphore(3)

    async def fan_out():
        llm = _FakeLLM()  # the ONE client the caller constructs for the whole fan-out
        return await asyncio.gather(*(history._distill(s, semaphore, llm) for s in sessions))

    results = asyncio.run(fan_out())
    assert len(results) == 3
    assert construction_count == 1


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
    assert captured.get("timeout") == common.SERVICE_TIMEOUT_SECONDS
