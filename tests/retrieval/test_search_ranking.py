"""search() end to end over a fake database and a fake reranker: what reaches the
reranker and what comes back. No DB, no vLLM."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager

import httpx
import pytest

from memory_base.retrieval import search

DAY = 86400.0


def memory_row(name: str, ts: float, text: str | None = None) -> dict:
    return {
        "id": name,
        "source_ref": name,
        "chunk_kind": "note",
        "metadata": {},
        "distilled": text or f"{name} body",
        "content_raw": text or f"{name} body",
        "ts_last_active": ts,
        "archived_at": None,
        "namespace": "default",
        "session_id": name,
    }


def code_row(name: str, code: str, start_line: int = 1) -> dict:
    return {
        "id": name,
        "repo": "repo",
        "filename": f"{name}.py",
        "code": code,
        "start_line": start_line,
        "end_line": start_line + 10,
        "mtime": time.time(),
    }


class FakeConn:
    def __init__(self, memory_vec=(), code_vec=(), neighbours=()):
        self.memory_vec = list(memory_vec)
        self.code_vec = list(code_vec)
        self.neighbours = list(neighbours)

    async def fetch(self, sql: str, *args):
        if "WHERE filename = $1" in sql:
            return self.neighbours
        if "to_bm25query" in sql:
            return []
        if "code_chunks" in sql:
            return self.code_vec
        return self.memory_vec


class FakeReranker:
    """Scores each document by the first configured marker it contains."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.payloads: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.payloads.append(payload)
        results = [
            {
                "index": i,
                "relevance_score": next(
                    (score for marker, score in self.scores.items() if marker in doc), 0.0
                ),
            }
            for i, doc in enumerate(payload["documents"])
        ]
        return httpx.Response(200, json={"results": results})

    @property
    def documents(self) -> list[str]:
        return self.payloads[-1]["documents"]


@pytest.fixture()
def run_search(monkeypatch):
    monkeypatch.setenv("RERANK_URL", "http://rerank.test")
    monkeypatch.setenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")

    async def embed(query):
        return "[0]"

    monkeypatch.setattr(search, "_embed_query", embed)

    def run(conn: FakeConn, reranker: FakeReranker, query: str = "q", **options):
        @asynccontextmanager
        async def acquire(*args, **kwargs):
            yield conn

        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(reranker.handler)
            return real_client(*args, **kwargs)

        monkeypatch.setattr(search.db, "acquire", acquire)
        monkeypatch.setattr(httpx, "AsyncClient", client)
        return asyncio.run(search.search(query, **options))

    return run


# ---- no age decay before the reranker --------------------------------------


def test_an_old_memory_with_a_strong_vector_rank_reaches_the_reranker(run_search):
    now = time.time()
    old = memory_row("old-note", now - 400 * DAY)
    fresh = [memory_row(f"fresh-{i:02d}", now - i * 60) for i in range(30)]
    reranker = FakeReranker({"old-note": 0.9})

    hits = run_search(FakeConn(memory_vec=[old, *fresh]), reranker, source="memory")

    assert any("old-note body" in doc for doc in reranker.documents)
    assert hits[0].meta["id"] == "old-note"


def test_the_multiplicative_age_decay_is_gone():
    assert not hasattr(search, "TIME_DECAY_HALF_LIFE_DAYS")
    assert not hasattr(search, "_apply_time_decay")
    assert not hasattr(search, "_decay_targets")


def test_the_longmemeval_manifest_records_no_decay_half_life():
    from memory_base.eval import longmemeval

    assert "TIME_DECAY_HALF_LIFE_DAYS" not in longmemeval._retrieval_constants()
