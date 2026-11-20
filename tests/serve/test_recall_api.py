"""Unit tests for POST /recall (memory_base.serve.search.routes + recall).

No DB and no upstream: the note search, the reranker and the decision store are
faked in the recall module, the way tests/serve/test_search_unavailable.py fakes search.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.retrieval.search import Hit, UpstreamUnavailable
from memory_base.serve import api
from memory_base.serve.decisions import store as decision_store
from memory_base.serve.search import recall

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})

SCOPE = "repo:github.com/o/r"


def _hit(ref, score):
    meta = {"id": ref, "kind": "work", "tags": [], "occurred_at": None}
    return Hit(ref=ref, text=f"note {ref}", ts=1_700_000_000.0, rerank_score=score, meta=meta)


def _decision(number, **overrides):
    row = {
        "number": number,
        "title": f"title {number}",
        "decision": f"decision {number}",
        "why": f"why {number}",
        "links": [],
        "decided_on": "2026-05-01",
    }
    return {**row, **overrides}


@pytest.fixture
def fakes(monkeypatch):
    state = {"search": [], "rerank": [], "decisions": [], "scores": {}, "notes": []}

    async def fake_search(query, **kwargs):
        state["search"].append((query, kwargs))
        return [h for h in state["notes"] if h.score >= kwargs["min_score"]]

    async def fake_rerank(query, texts):
        state["rerank"].append((query, texts))
        return [state["scores"][text.split("\n")[0]] for text in texts]

    async def fake_active(**kwargs):
        state["decisions_call"] = kwargs
        return state["decisions"]

    monkeypatch.setattr(recall, "search", fake_search)
    monkeypatch.setattr(recall, "rerank_scores", fake_rerank)
    monkeypatch.setattr(decision_store, "active_decisions", fake_active)
    return state


def _post(**body):
    return client.post("/recall", json={"query": "which database", **body})


def test_notes_and_decisions_are_returned_as_separate_scored_lists(fakes):
    fakes["notes"] = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7), _hit("d", 0.6)]
    fakes["decisions"] = [_decision(3), _decision(2), _decision(1)]
    fakes["scores"] = {"D-3 title 3": 0.95, "D-2 title 2": 0.4, "D-1 title 1": 0.7}
    response = _post(scope=SCOPE)
    assert response.status_code == 200
    body = response.json()
    assert [n["id"] for n in body["notes"]] == ["a", "b", "c"]
    assert body["notes"][0]["score"] == 0.9
    assert [d["number"] for d in body["decisions"]] == [3, 1]
    assert body["decisions"][0] == {
        "number": 3,
        "title": "title 3",
        "decided_on": "2026-05-01",
        "decision": "decision 3",
        "why": "why 3",
        "links": [],
        "score": 0.95,
    }
    assert body["errors"] == []


def test_the_note_search_runs_as_the_caller_with_the_floor(fakes):
    _post(notes={"min_score": 0.6, "top_k": 5}, scope=SCOPE)
    query, kwargs = fakes["search"][0]
    assert query == "which database"
    assert kwargs["visible_to"] == "claude-code"
    assert kwargs["min_score"] == 0.6


def test_the_note_search_gets_the_last_two_thousand_chars_and_decisions_the_whole_query(fakes):
    fakes["decisions"] = [_decision(1)]
    fakes["scores"] = {"D-1 title 1": 0.9}
    query = "h" * 2000 + "t" * 2000
    client.post("/recall", json={"query": query, "scope": SCOPE})
    assert fakes["search"][0][0] == "t" * 2000
    assert fakes["rerank"][0][0] == query


def test_decisions_are_the_callers_newest_two_hundred_and_texts_carry_the_number(fakes):
    fakes["decisions"] = [_decision(2), _decision(1, why=None)]
    fakes["scores"] = {"D-2 title 2": 0.9, "D-1 title 1": 0.9}
    _post(scope=SCOPE)
    assert fakes["decisions_call"] == {"author": "claude-code", "scope": SCOPE, "limit": 200}
    assert fakes["rerank"][0][1] == [
        "D-2 title 2\n\ndecision 2\n\nwhy 2",
        "D-1 title 1\n\ndecision 1",
    ]


def test_the_decision_floor_and_top_k_apply_to_the_reranked_score(fakes):
    fakes["decisions"] = [_decision(n) for n in (4, 3, 2, 1)]
    fakes["scores"] = {
        "D-4 title 4": 0.9,
        "D-3 title 3": 0.8,
        "D-2 title 2": 0.7,
        "D-1 title 1": 0.6,
    }
    body = _post(scope=SCOPE, decisions={"top_k": 3, "min_score": 0.75}).json()
    assert [d["number"] for d in body["decisions"]] == [4, 3]
    body = _post(scope=SCOPE, decisions={"top_k": 3, "min_score": 0.0}).json()
    assert [d["number"] for d in body["decisions"]] == [4, 3, 2]


def test_equal_scores_order_by_number_newest_first(fakes):
    fakes["decisions"] = [_decision(n) for n in (5, 4, 3)]
    fakes["scores"] = {"D-5 title 5": 0.8, "D-4 title 4": 0.8, "D-3 title 3": 0.8}
    body = _post(scope=SCOPE, decisions={"top_k": 2}).json()
    assert [d["number"] for d in body["decisions"]] == [5, 4]


def test_no_active_decision_means_no_rerank_call(fakes):
    fakes["decisions"] = []
    body = _post(scope=SCOPE).json()
    assert body["decisions"] == [] and body["errors"] == []
    assert fakes["rerank"] == []


def test_without_a_scope_the_decision_lane_is_not_read(fakes):
    fakes["notes"] = [_hit("a", 0.9)]
    body = _post().json()
    assert body["decisions"] == [] and len(body["notes"]) == 1
    assert "decisions_call" not in fakes and fakes["rerank"] == []


def test_one_failing_branch_keeps_the_other_and_names_the_failed_one(monkeypatch, fakes):
    fakes["notes"] = [_hit("a", 0.9)]

    async def broken(**kwargs):
        raise UpstreamUnavailable("reranking")

    monkeypatch.setattr(decision_store, "active_decisions", broken)
    response = _post(scope=SCOPE)
    assert response.status_code == 200
    assert [n["id"] for n in response.json()["notes"]] == ["a"]
    assert response.json()["decisions"] == []
    assert response.json()["errors"] == ["decisions"]

    async def broken_search(query, **kwargs):
        raise UpstreamUnavailable("embedding")

    async def fake_active_again(**kwargs):
        return [_decision(1)]

    monkeypatch.setattr(recall, "search", broken_search)
    monkeypatch.setattr(decision_store, "active_decisions", fake_active_again)
    fakes["scores"] = {"D-1 title 1": 0.9}
    body = _post(scope=SCOPE).json()
    assert body["notes"] == [] and body["errors"] == ["notes"]
    assert [d["number"] for d in body["decisions"]] == [1]


def test_both_branches_failing_answers_503_in_the_search_log_style(monkeypatch, fakes, caplog):
    async def broken_search(query, **kwargs):
        raise UpstreamUnavailable("embedding") from TimeoutError("connect stalled")

    async def broken_active(**kwargs):
        raise UpstreamUnavailable("reranking")

    monkeypatch.setattr(recall, "search", broken_search)
    monkeypatch.setattr(decision_store, "active_decisions", broken_active)
    with caplog.at_level("WARNING"):
        response = _post(scope=SCOPE)
    assert response.status_code == 503
    assert "memory cannot be attached right now" in response.json()["error"]
    lines = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("notes" in x and "embedding" in x and "connect stalled" in x for x in lines)
    assert any("decisions" in x and "reranking" in x and "elapsed" in x for x in lines)


def test_a_failing_branch_alone_is_a_503_when_it_is_the_only_one_requested(monkeypatch, fakes):
    async def broken_search(query, **kwargs):
        raise UpstreamUnavailable("embedding")

    monkeypatch.setattr(recall, "search", broken_search)
    assert _post().status_code == 503


@pytest.mark.parametrize(
    "overrides",
    [
        {"query": ""},
        {"query": "  "},
        {"query": "q" * 4001},
        {"query": 5},
        {"scope": "project:o/p"},
        {"scope": "repo:localhost/o/r"},
        {"scope": 5},
        {"notes": {"top_k": 6}},
        {"notes": {"top_k": 0}},
        {"decisions": {"top_k": True}},
        {"notes": {"min_score": 1.5}},
        {"decisions": {"min_score": "high"}},
        {"notes": [1]},
        {"unknown": 1},
    ],
)
def test_bad_bodies_answer_400(fakes, overrides):
    assert client.post("/recall", json={"query": "q", **overrides}).status_code == 400


def test_a_query_of_exactly_four_thousand_chars_is_accepted(fakes):
    assert client.post("/recall", json={"query": "q" * 4000}).status_code == 200
