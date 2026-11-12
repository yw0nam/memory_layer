"""Unit tests for the REST API server (memory_base.serve.api).

No DB, no network: routes only parse/validate/delegate/JSON, so the hooks a route calls into are monkeypatched
at the module level, matching the existing convention (e.g.
``monkeypatch.setattr(answer, "search", fake_search)`` in test_answer_mcp.py):

- ``search_routes.search``  -- delegate for POST /search (memory_base.retrieval.search.search)
- ``store.save_note``  -- delegate for POST /save_memory (memory_base.serve.notes.store)
- ``health.db_healthy``, ``health.embedding_healthy``, ``health.rerank_healthy``
  -- async dependency probes used by GET /health/services
- ``search_routes.hit_to_dict``  -- Hit -> {id, ref, date, score, text, ...} serializer
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.retrieval.search import Hit
from memory_base.serve import api, health
from memory_base.serve.search import routes as search_routes
from memory_base.serve.notes import store

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})


def _hit(ref="save_memory", text="body", rrf=0.5, rerank_score=None):
    return Hit(
        ref=ref,
        text=text,
        ts=1_700_000_000.0,
        rrf=rrf,
        rerank_score=rerank_score,
        meta={"id": f"note:{ref}", "kind": "work", "tags": [], "occurred_at": None},
    )


# ---- GET /health and /health/services -------------------------------------------------------


def test_health_is_200_without_reaching_any_dependency(monkeypatch):
    """The container healthcheck curls this, so a dead backend must not fail it."""

    async def unreachable():
        raise AssertionError("liveness must not probe a dependency")

    for probe in ("db_healthy", "embedding_healthy", "rerank_healthy"):
        monkeypatch.setattr(health, probe, unreachable)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.fixture()
def _all_probes_up(monkeypatch):
    """Default every dependency probe to healthy; tests override individual ones."""

    async def up():
        return True

    monkeypatch.setattr(health, "db_healthy", up)
    monkeypatch.setattr(health, "embedding_healthy", up)
    monkeypatch.setattr(health, "rerank_healthy", up)


def test_health_ok_when_all_dependencies_reachable(_all_probes_up):
    response = client.get("/health/services")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"db": True, "embedding": True, "rerank": True},
    }


def test_health_503_when_db_unreachable(_all_probes_up, monkeypatch):
    async def fake_db_healthy():
        raise ConnectionError("db down")

    monkeypatch.setattr(health, "db_healthy", fake_db_healthy)
    response = client.get("/health/services")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["db"] is False


def test_health_503_when_embedding_unreachable(_all_probes_up, monkeypatch):
    async def fake_embedding_healthy():
        return False

    monkeypatch.setattr(health, "embedding_healthy", fake_embedding_healthy)
    response = client.get("/health/services")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["embedding"] is False


def test_health_503_when_rerank_unreachable(_all_probes_up, monkeypatch):
    async def fake_rerank_healthy():
        return False

    monkeypatch.setattr(health, "rerank_healthy", fake_rerank_healthy)
    response = client.get("/health/services")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["rerank"] is False


# ---- POST /search --------------------------------------------------------


def test_search_empty_query_400():
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 400
    assert "error" in response.json()


@pytest.mark.parametrize(
    "body",
    [
        {"query": "hello", "kind": "reminder"},
        {"query": "hello", "tags": []},
        {"query": "hello", "tags": None},
        {"query": "hello", "tags": "infra"},
    ],
)
def test_search_filter_validation_errors_are_400(body):
    response = client.post("/search", json=body)
    assert response.status_code == 400
    assert set(response.json()) == {"error"}


def test_search_forwards_raw_kind_and_tags(monkeypatch):
    """The route forwards raw body values as-is; search() does the normalizing."""
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post(
        "/search",
        json={
            "query": "hello",
            "kind": "work",
            "tags": [" Infra ", "DATABASE", "infra"],
        },
    )
    assert response.status_code == 200
    assert captured["kind"] == "work"
    assert captured["tags"] == [" Infra ", "DATABASE", "infra"]


def test_search_silently_ignores_include_atoms(monkeypatch):
    """No compat shim: a caller still sending include_atoms is accepted and ignored."""
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post(
        "/search",
        json={"query": "hello", "include_atoms": False},
    )
    assert response.status_code == 200
    assert "include_atoms" not in captured


def test_search_forwards_namespaces_filter(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post(
        "/search", json={"query": "hello", "namespaces": [" team-a ", "team-a", "team-b"]}
    )
    assert response.status_code == 200
    assert captured["namespaces"] == ["team-a", "team-b"]


def test_search_omitted_namespaces_does_not_reach_search(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert "namespaces" not in captured


def test_search_malformed_namespaces_400():
    response = client.post("/search", json={"query": "hello", "namespaces": "team-a"})
    assert response.status_code == 400
    assert "namespaces" in response.json()["error"]


def test_search_returns_hit_to_dict_shape(monkeypatch):
    hits = [_hit(text="x", rrf=0.5, rerank_score=0.9)]

    async def fake_search(query, include_archived=False, **options):
        return hits

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body == [search_routes.hit_to_dict(h) for h in hits]
    assert set(body[0]) >= {"id", "ref", "date", "score", "text"}


def test_search_default_top_k_is_10(monkeypatch):
    hits = [_hit(ref=f"note-{i}", rrf=float(i)) for i in range(15)]

    async def fake_search(query, include_archived=False, **options):
        return hits

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert len(response.json()) == 10


def test_search_respects_custom_top_k(monkeypatch):
    hits = [_hit(ref=f"note-{i}", rrf=float(i)) for i in range(5)]

    async def fake_search(query, include_archived=False, **options):
        return hits

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello", "top_k": 2})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_search_omitted_min_score_does_not_reach_search(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert "min_score" not in captured


@pytest.mark.parametrize("min_score", [True, "0.7", -0.1, 1.5])
def test_search_invalid_min_score_400(min_score):
    response = client.post("/search", json={"query": "hello", "min_score": min_score})
    assert response.status_code == 400
    assert response.json()["error"] == "min_score must be a number between 0 and 1"


@pytest.mark.parametrize("min_score", [0, 1])
def test_search_boundary_min_score_accepted(monkeypatch, min_score):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello", "min_score": min_score})
    assert response.status_code == 200
    assert captured["min_score"] == min_score


def test_search_malformed_json_400():
    response = client.post(
        "/search", content=b"{not valid json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert "error" in response.json()


# ---- POST /save_memory ----------------------------------------------------


def test_save_memory_empty_content_400():
    response = client.post("/save_memory", json={"content": "", "kind": "work"})
    assert response.status_code == 400
    assert response.json()["error"] == "content must not be empty"


def test_save_memory_oversized_content_400():
    response = client.post("/save_memory", json={"content": "x" * 1751, "kind": "work"})
    assert response.status_code == 400
    assert response.json()["error"] == "content exceeds 1750 chars"


def test_save_memory_bad_kind_400():
    response = client.post("/save_memory", json={"content": "valid content", "kind": "reminder"})
    assert response.status_code == 400
    assert response.json()["error"] == "kind must be one of ('personal', 'work')"


@pytest.mark.parametrize("tags", ["infra", [1], ["infra", None]])
def test_save_memory_malformed_tags_400(tags):
    response = client.post(
        "/save_memory",
        json={"content": "valid content", "kind": "work", "tags": tags},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "tags must be a non-empty list of strings"


def test_save_memory_missing_tags_400():
    response = client.post("/save_memory", json={"content": "valid content", "kind": "work"})
    assert response.status_code == 400
    assert response.json()["error"] == "tags must be a non-empty list of strings"


def test_save_memory_valid_content_delegates_to_save_note(monkeypatch):
    captured = {}

    async def fake_save_note(
        content,
        *,
        tags,
        kind,
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
        allow_similar=False,
        shared=False,
    ):
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

    monkeypatch.setattr(store, "save_note", fake_save_note)
    response = client.post("/save_memory", json={"content": "distilled note text", "kind": "work"})
    assert response.status_code == 200
    assert response.json() == {
        "id": "note:deadbeefdeadbeef",
        "kind": "work",
        "stored": True,
        "superseded": None,
        "similar": [],
    }
    assert captured == {
        "content": "distilled note text",
        "kind": "work",
        "tags": None,
        "supersedes": None,
    }


def test_save_memory_malformed_json_400():
    response = client.post(
        "/save_memory", content=b"{not valid json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_save_memory_omitted_namespace_defaults_to_default(monkeypatch):
    captured = {}

    async def fake_save_note(
        content,
        *,
        tags,
        kind,
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
        allow_similar=False,
        shared=False,
    ):
        captured["namespace"] = namespace
        return {"id": "note:x", "kind": kind, "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(store, "save_note", fake_save_note)
    response = client.post("/save_memory", json={"content": "distilled note text", "kind": "work"})
    assert response.status_code == 200
    assert captured["namespace"] == "default"


def test_save_memory_forwards_explicit_namespace(monkeypatch):
    captured = {}

    async def fake_save_note(
        content,
        *,
        tags,
        kind,
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
        allow_similar=False,
        shared=False,
    ):
        captured["namespace"] = namespace
        return {"id": "note:x", "kind": kind, "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(store, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={
            "content": "distilled note text",
            "kind": "work",
            "namespace": "team-a",
        },
    )
    assert response.status_code == 200
    assert captured["namespace"] == "team-a"


def test_save_memory_unregistered_namespace_400(monkeypatch):
    async def fake_save_note(
        content,
        *,
        tags,
        kind,
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
        allow_similar=False,
        shared=False,
    ):
        raise ValueError(f"unregistered namespace: {namespace}")

    monkeypatch.setattr(store, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={
            "content": "distilled note text",
            "kind": "work",
            "namespace": "ghost",
        },
    )
    assert response.status_code == 400
    assert "unregistered namespace" in response.json()["error"]


@pytest.mark.parametrize("namespace", ["   ", None])
def test_save_memory_malformed_namespace_400(namespace):
    response = client.post(
        "/save_memory",
        json={
            "content": "distilled note text",
            "kind": "work",
            "namespace": namespace,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "namespace must be a non-empty string"


# ---- source rename: history rejected ----------------------------------------


# ---- POST /search since/until ----------------------------------------------


def test_search_forwards_raw_since_and_until(monkeypatch):
    """The route forwards raw body values as-is; search() does the parsing."""
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post(
        "/search",
        json={"query": "q", "since": "2026-08-01", "until": "2026-08-12"},
    )
    assert response.status_code == 200
    assert captured["since"] == "2026-08-01"
    assert captured["until"] == "2026-08-12"


def test_search_omitted_time_bounds_do_not_reach_search(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "q", "source": "memory"})
    assert response.status_code == 200
    assert "since" not in captured
    assert "until" not in captured


@pytest.mark.parametrize("field", ["since", "until"])
def test_search_explicit_null_time_bound_400(field):
    response = client.post("/search", json={"query": "q", field: None})
    assert response.status_code == 400
    assert field in response.json()["error"]


@pytest.mark.parametrize(
    "body",
    [
        {"query": "q", "since": "not-a-date"},
        {"query": "q", "since": "2026-08-13", "until": "2026-08-12"},
    ],
)
def test_search_time_bound_validation_errors_are_400(body):
    response = client.post("/search", json=body)
    assert response.status_code == 400
    assert set(response.json()) == {"error"}


def test_search_forwards_budget_tokens_and_ignores_top_k(monkeypatch):
    captured = {}
    hits = [_hit(ref=f"note-{i}", rrf=float(i)) for i in range(15)]

    async def fake_search(query, **options):
        captured.update(options)
        return hits

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello", "top_k": 2, "budget_tokens": 800})
    assert response.status_code == 200
    assert captured["budget_tokens"] == 800
    assert len(response.json()) == 15


def test_search_omitted_budget_tokens_does_not_reach_search(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert "budget_tokens" not in captured


@pytest.mark.parametrize("budget_tokens", [True, 1.5, 0, 32001])
def test_search_invalid_budget_tokens_400(budget_tokens):
    response = client.post("/search", json={"query": "hello", "budget_tokens": budget_tokens})
    assert response.status_code == 400
    assert response.json()["error"] == "budget_tokens must be an integer between 1 and 32000"
