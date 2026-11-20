"""Unit tests for the /decisions REST endpoints (memory_base.serve.decisions.routes).

No DB: validation runs before storage, and the store functions are monkeypatched
where a route's delegation is under test.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.decisions import store

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})

SCOPE = "repo:github.com/o/r"
VALID = {"scope": SCOPE, "title": "Use Postgres", "decision": "Postgres holds it."}


def _unreachable(captured):
    async def fake(**kwargs):
        captured.append(kwargs)
        raise AssertionError("storage must not be reached")

    return fake


def test_record_delegates_with_the_header_author(monkeypatch):
    captured = {}

    async def fake_record(**kwargs):
        captured.update(kwargs)
        return {"number": 1, "superseded": None}

    monkeypatch.setattr(store, "record_decision", fake_record)
    response = client.post(
        "/decisions",
        json={**VALID, "why": "pgvector", "links": ["https://x.y/1"], "decided_on": "2026-05-01"},
        headers={"X-Author": "natsume"},
    )
    assert response.status_code == 201
    assert response.json() == {"number": 1, "superseded": None}
    assert captured["author"] == "natsume"
    assert captured["scope"] == SCOPE
    assert captured["why"] == "pgvector"
    assert captured["decided_on"] == "2026-05-01"


@pytest.mark.parametrize("path", ["/decisions", "/decisions/retire"])
def test_a_body_carrying_author_is_rejected(monkeypatch, path):
    captured = []
    monkeypatch.setattr(store, "record_decision", _unreachable(captured))
    monkeypatch.setattr(store, "retire_decision", _unreachable(captured))
    response = client.post(path, json={**VALID, "number": 1, "reason": "r", "author": "x"})
    assert response.status_code == 400
    assert "X-Author" in response.json()["error"]
    assert not captured


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": ""},
        {"title": "t" * 201},
        {"decision": " "},
        {"decision": "d" * 1000, "why": "w" * 501},
        {"links": ["https://x.y"] * 11},
        {"links": ["l" * 501]},
        {"links": "https://x.y"},
        {"decided_on": "yesterday"},
        {"decided_on": "20260501"},
        {"decided_on": "2026-W18-5"},
        {"scope": "project:o/p"},
        {"scope": "repo:localhost/o/r"},
        {"supersedes": "D-1"},
        {"unknown": 1},
    ],
)
def test_record_limits_answer_400(overrides):
    response = client.post("/decisions", json={**VALID, **overrides})
    assert response.status_code == 400


def test_a_refused_supersede_answers_400(monkeypatch):
    async def fake_record(**kwargs):
        raise ValueError("D-3 is not an active decision of this scope and author")

    monkeypatch.setattr(store, "record_decision", fake_record)
    response = client.post("/decisions", json={**VALID, "supersedes": 3})
    assert response.status_code == 400
    assert "D-3" in response.json()["error"]


def test_retire_delegates_and_a_refusal_answers_400(monkeypatch):
    captured = {}

    async def fake_retire(**kwargs):
        captured.update(kwargs)
        if kwargs["number"] == 9:
            raise ValueError("D-9 is not an active decision of this scope and author")
        return {"number": kwargs["number"], "status": "retired"}

    monkeypatch.setattr(store, "retire_decision", fake_retire)
    ok = client.post("/decisions/retire", json={"scope": SCOPE, "number": 2, "reason": "moved"})
    assert ok.status_code == 200
    assert ok.json() == {"number": 2, "status": "retired"}
    assert captured["author"] == "claude-code"
    assert captured["reason"] == "moved"
    refused = client.post("/decisions/retire", json={"scope": SCOPE, "number": 9, "reason": "x"})
    assert refused.status_code == 400


@pytest.mark.parametrize("body", [{"scope": SCOPE, "number": "2", "reason": "r"}, {"scope": SCOPE}])
def test_retire_validates_its_body(body):
    assert client.post("/decisions/retire", json=body).status_code == 400


def test_list_defaults_to_active_and_caps_the_limit(monkeypatch):
    captured = {}

    async def fake_list(**kwargs):
        captured.update(kwargs)
        return [{"number": 2, "title": "b", "decided_on": "2026-05-02", "status": "active"}]

    monkeypatch.setattr(store, "list_decisions", fake_list)
    response = client.get("/decisions", params={"scope": SCOPE})
    assert response.status_code == 200
    assert response.json()[0]["number"] == 2
    assert captured == {"author": "claude-code", "scope": SCOPE, "status": "active", "limit": 200}
    client.get("/decisions", params={"scope": SCOPE, "status": "all", "limit": "5"})
    assert captured["status"] == "all" and captured["limit"] == 5


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"scope": SCOPE, "limit": "201"},
        {"scope": SCOPE, "limit": "0"},
        {"scope": SCOPE, "limit": "x"},
        {"scope": SCOPE, "status": "retired"},
        {"scope": "project:o/p"},
    ],
)
def test_list_rejects_bad_params(params):
    assert client.get("/decisions", params=params).status_code == 400


def test_get_returns_the_full_row_and_404_for_an_unknown_number(monkeypatch):
    row = {"number": 2, "title": "t", "supersedes": 1, "superseded_by": None}

    async def fake_get(**kwargs):
        store.repo_scope(kwargs["scope"])
        if kwargs["number"] != 2:
            raise store.DecisionNotFound("no decision D-%d" % kwargs["number"])
        return row

    monkeypatch.setattr(store, "get_decision", fake_get)
    assert client.get("/decisions/2", params={"scope": SCOPE}).json() == row
    assert client.get("/decisions/7", params={"scope": SCOPE}).status_code == 404
    assert client.get("/decisions/x", params={"scope": SCOPE}).status_code == 400
    assert client.get("/decisions/2").status_code == 400


@pytest.mark.parametrize("missing", ["scope", "title", "decision"])
def test_record_without_a_required_field_answers_400(missing):
    body = {k: v for k, v in VALID.items() if k != missing}
    assert client.post("/decisions", json=body).status_code == 400
