"""Unit tests for the REST API server (memory_base.serve.api).

No DB, no network: routes only parse/validate/delegate/JSON per
docs/specs/impl_rest_api.md, so the hooks a route calls into are monkeypatched
at the module level, matching the existing convention (e.g.
``monkeypatch.setattr(answer, "search", fake_search)`` in test_answer_mcp.py):

- ``api.search``       -- delegate for POST /search (memory_base.retrieval.search.search)
- ``api.save_note``    -- delegate for POST /save_memory (memory_base.serve.notes.save_note)
- ``api.log_retrieval``-- best-effort access logging after a search (memory_base.serve.access_log)
- ``api.db_healthy``   -- async DB connectivity check used by GET /health
- ``api.hit_to_dict``  -- Hit -> {source, ref, date, score, text[, context]} serializer

Collection fails today: memory_base.serve.api does not exist yet.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.retrieval.search import Hit
from memory_base.serve import api

client = TestClient(api.app)


def _hit(
    source="code",
    ref="a.py:L1-L2",
    text="body",
    ts=1_700_000_000.0,
    rrf=0.5,
    rerank_score=None,
    meta=None,
):
    return Hit(
        source=source,
        ref=ref,
        text=text,
        ts=ts,
        rrf=rrf,
        rerank_score=rerank_score,
        meta=meta or {},
    )


@pytest.fixture(autouse=True)
def _no_op_access_log(monkeypatch):
    async def fake_log_retrieval(query, source, hits):
        return None

    monkeypatch.setattr(api, "log_retrieval", fake_log_retrieval)


# ---- GET /health -------------------------------------------------------


def test_health_ok_when_db_reachable(monkeypatch):
    async def fake_db_healthy():
        return None

    monkeypatch.setattr(api, "db_healthy", fake_db_healthy)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_503_when_db_unreachable(monkeypatch):
    async def fake_db_healthy():
        raise ConnectionError("db down")

    monkeypatch.setattr(api, "db_healthy", fake_db_healthy)
    response = client.get("/health")
    assert response.status_code == 503


# ---- POST /search --------------------------------------------------------


def test_search_missing_query_400():
    response = client.post("/search", json={})
    assert response.status_code == 400
    assert "error" in response.json()


def test_search_empty_query_400():
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 400
    assert "error" in response.json()


def test_search_invalid_source_400():
    response = client.post("/search", json={"query": "hello", "source": "bogus"})
    assert response.status_code == 400
    assert "error" in response.json()


def test_search_returns_hit_to_dict_shape(monkeypatch):
    hits = [_hit(source="code", ref="a.py:L1-L2", text="x", rrf=0.5, rerank_score=0.9)]

    async def fake_search(query, source="all", include_archived=False):
        return hits

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body == [api.hit_to_dict(h) for h in hits]
    assert set(body[0]) >= {"source", "ref", "date", "score", "text"}


def test_search_defaults_source_to_all(monkeypatch):
    captured = {}

    async def fake_search(query, source="all", include_archived=False):
        captured["query"] = query
        captured["source"] = source
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert captured["query"] == "hello"
    assert captured["source"] == "all"


def test_search_passes_requested_source_through(monkeypatch):
    captured = {}

    async def fake_search(query, source="all", include_archived=False):
        captured["source"] = source
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello", "source": "code"})
    assert response.status_code == 200
    assert captured["source"] == "code"


def test_search_default_top_k_is_10(monkeypatch):
    hits = [_hit(ref=f"f{i}.py:L1-L2", rrf=float(i)) for i in range(15)]

    async def fake_search(query, source="all", include_archived=False):
        return hits

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert len(response.json()) == 10


def test_search_respects_custom_top_k(monkeypatch):
    hits = [_hit(ref=f"f{i}.py:L1-L2", rrf=float(i)) for i in range(5)]

    async def fake_search(query, source="all", include_archived=False):
        return hits

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello", "top_k": 2})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_search_malformed_json_400():
    response = client.post(
        "/search", content=b"{not valid json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert "error" in response.json()


# ---- POST /save_memory ----------------------------------------------------


def test_save_memory_empty_content_400():
    response = client.post("/save_memory", json={"content": ""})
    assert response.status_code == 400
    assert response.json()["error"] == "content must not be empty"


def test_save_memory_oversized_content_400():
    response = client.post("/save_memory", json={"content": "x" * 4001})
    assert response.status_code == 400
    assert response.json()["error"] == "content exceeds 4000 chars"


def test_save_memory_bad_kind_400():
    response = client.post("/save_memory", json={"content": "valid content", "kind": "reminder"})
    assert response.status_code == 400
    assert response.json()["error"] == "kind must be one of ('note', 'decision')"


def test_save_memory_valid_content_delegates_to_save_note(monkeypatch):
    captured = {}

    async def fake_save_note(content, kind="note", tags=None, supersedes=None):
        captured["content"] = content
        captured["kind"] = kind
        captured["tags"] = tags
        captured["supersedes"] = supersedes
        return {
            "id": "note:deadbeefdeadbeef",
            "kind": kind,
            "stored": True,
            "superseded": None,
            "similar": [],
        }

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post("/save_memory", json={"content": "distilled note text"})
    assert response.status_code == 200
    assert response.json() == {
        "id": "note:deadbeefdeadbeef",
        "kind": "note",
        "stored": True,
        "superseded": None,
        "similar": [],
    }
    assert captured == {
        "content": "distilled note text",
        "kind": "note",
        "tags": None,
        "supersedes": None,
    }


def test_save_memory_malformed_json_400():
    response = client.post(
        "/save_memory", content=b"{not valid json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert "error" in response.json()
