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
        "conversation_id": None,
        "source_turn_start": None,
        "source_turn_end": None,
        "occurred_at": None,
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


def test_forty_fused_candidates_reach_the_reranker(run_search):
    now = time.time()
    rows = [memory_row(f"cand-{i:02d}", now - i * 60) for i in range(45)]
    reranker = FakeReranker({})

    run_search(FakeConn(memory_vec=rows), reranker, source="memory")

    assert len(reranker.documents) == 40
    assert all(any(f"cand-{i:02d} body" in d for d in reranker.documents) for i in range(40))


def test_the_multiplicative_age_decay_is_gone():
    assert not hasattr(search, "TIME_DECAY_HALF_LIFE_DAYS")
    assert not hasattr(search, "_apply_time_decay")
    assert not hasattr(search, "_decay_targets")


def test_the_longmemeval_manifest_records_no_decay_half_life():
    from memory_base.eval import longmemeval

    assert "TIME_DECAY_HALF_LIFE_DAYS" not in longmemeval._retrieval_constants()


# ---- token-budget packing --------------------------------------------------


def _scored_rows(count: int, chars: int) -> tuple[list[dict], dict[str, float]]:
    now = time.time()
    rows = [
        memory_row(f"note-{i:02d}", now - i * 60, f"note-{i:02d} " + "x" * (chars - 8))
        for i in range(count)
    ]
    # Every other hit scores below MIN_SCORE, which budget mode ignores.
    scores = {f"note-{i:02d}": 0.9 - i * 0.01 - (0.8 if i % 2 else 0.0) for i in range(count)}
    return rows, scores


def test_a_budget_returns_more_than_ten_short_hits_in_reranked_order(run_search):
    rows, scores = _scored_rows(15, 40)
    reranker = FakeReranker(scores)

    hits = run_search(
        FakeConn(memory_vec=rows), reranker, source="memory", budget_tokens=1000, min_score=0.5
    )

    assert len(hits) == 15 > search.RERANK_TOP
    assert [h.rerank_score for h in hits] == sorted(scores.values(), reverse=True)


def test_a_budget_stops_before_the_hit_that_would_exceed_it(run_search):
    rows, scores = _scored_rows(6, 400)

    hits = run_search(
        FakeConn(memory_vec=rows), FakeReranker(scores), source="memory", budget_tokens=250
    )

    assert [h.meta["id"] for h in hits] == ["note-00", "note-02"]
    assert sum(search.estimate_tokens(h) for h in hits) <= 250


def test_a_budget_counts_a_code_hit_with_its_restored_context(run_search):
    now = time.time()
    first = memory_row("first-note", now, "first-note " + "x" * 389)
    last = memory_row("last-note", now, "last-note " + "x" * 70)
    code = code_row("module", "def f(): pass  # module marker")
    neighbours = [{"code": "y" * 800, "start_line": 20}]
    reranker = FakeReranker({"first-note": 0.9, "module marker": 0.8, "last-note": 0.7})

    hits = run_search(
        FakeConn(memory_vec=[first, last], code_vec=[code], neighbours=neighbours),
        reranker,
        source="all",
        budget_tokens=150,
    )

    assert [h.ref for h in hits] == ["first-note"]


def test_the_token_estimate_is_a_quarter_of_the_characters_and_at_least_one():
    hit = search.Hit(source="memory", ref="r", text="x" * 41, ts=0.0)
    empty = search.Hit(source="memory", ref="r", text="", ts=0.0)
    with_context = search.Hit(
        source="code", ref="r", text="x" * 40, ts=0.0, meta={"context": "y" * 40}
    )

    assert search.estimate_tokens(hit) == 10
    assert search.estimate_tokens(empty) == 1
    assert search.estimate_tokens(with_context) == 20


def test_without_a_budget_rerank_top_and_the_floor_still_apply(run_search):
    rows, scores = _scored_rows(15, 40)

    hits = run_search(FakeConn(memory_vec=rows), FakeReranker(scores), source="memory")

    assert len(hits) <= search.RERANK_TOP
    assert all(h.score >= search.MIN_SCORE for h in hits)
    assert len(hits) == 8
