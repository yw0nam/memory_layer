"""Unit tests for the REST API server (memory_base.serve.api).

No DB, no network: routes only parse/validate/delegate/JSON, so the hooks a route calls into are monkeypatched
at the module level, matching the existing convention (e.g.
``monkeypatch.setattr(answer, "search", fake_search)`` in test_answer_mcp.py):

- ``api.search``       -- delegate for POST /search (memory_base.retrieval.search.search)
- ``api.save_note``    -- delegate for POST /save_memory (memory_base.serve.notes.save_note)
- ``api.access_log``   -- buffered access logging after a search (memory_base.serve.access_log)
- ``api.db_healthy``, ``api.embedding_healthy``, ``api.rerank_healthy``, ``api.llm_healthy``
  -- async dependency probes used by GET /health/services
- ``api.hit_to_dict``  -- Hit -> {source, ref, date, score, text[, context]} serializer

Collection fails today: memory_base.serve.api does not exist yet.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.retrieval.search import Hit
from memory_base.serve import api

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


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
def _empty_access_log_buffer(monkeypatch):
    monkeypatch.setattr(api.access_log, "_pending_logs", [])
    monkeypatch.setattr(api.access_log, "_pending_hits", {})


# ---- GET /health and /health/services -------------------------------------------------------


def test_health_is_200_without_reaching_any_dependency(monkeypatch):
    """The container healthcheck curls this, so a dead backend must not fail it."""

    async def unreachable():
        raise AssertionError("liveness must not probe a dependency")

    for probe in ("db_healthy", "embedding_healthy", "rerank_healthy", "llm_healthy"):
        monkeypatch.setattr(api, probe, unreachable)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.fixture()
def _all_probes_up(monkeypatch):
    """Default every dependency probe to healthy; tests override individual ones."""

    async def up():
        return True

    monkeypatch.setattr(api, "db_healthy", up)
    monkeypatch.setattr(api, "embedding_healthy", up)
    monkeypatch.setattr(api, "rerank_healthy", up)
    monkeypatch.setattr(api, "llm_healthy", up)


def test_health_ok_when_all_dependencies_reachable(_all_probes_up):
    response = client.get("/health/services")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"db": True, "embedding": True, "rerank": True, "llm": True},
    }


def test_health_503_when_db_unreachable(_all_probes_up, monkeypatch):
    async def fake_db_healthy():
        raise ConnectionError("db down")

    monkeypatch.setattr(api, "db_healthy", fake_db_healthy)
    response = client.get("/health/services")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["db"] is False


def test_health_503_when_embedding_unreachable(_all_probes_up, monkeypatch):
    async def fake_embedding_healthy():
        return False

    monkeypatch.setattr(api, "embedding_healthy", fake_embedding_healthy)
    response = client.get("/health/services")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["embedding"] is False


def test_health_503_when_rerank_unreachable(_all_probes_up, monkeypatch):
    async def fake_rerank_healthy():
        return False

    monkeypatch.setattr(api, "rerank_healthy", fake_rerank_healthy)
    response = client.get("/health/services")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["rerank"] is False


def test_health_200_when_only_llm_unreachable(_all_probes_up, monkeypatch):
    async def fake_llm_healthy():
        return False

    monkeypatch.setattr(api, "llm_healthy", fake_llm_healthy)
    response = client.get("/health/services")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"db": True, "embedding": True, "rerank": True, "llm": False}


def test_health_probe_raising_is_reported_as_false_not_propagated(_all_probes_up, monkeypatch):
    async def fake_llm_healthy():
        raise TimeoutError("connect timed out")

    monkeypatch.setattr(api, "llm_healthy", fake_llm_healthy)
    response = client.get("/health/services")
    assert response.status_code == 200
    assert response.json()["checks"]["llm"] is False


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


@pytest.mark.parametrize(
    "body",
    [
        {"query": "hello", "source": "all", "kind": "note"},
        {"query": "hello", "source": "code", "tags": ["infra"]},
        {"query": "hello", "source": "memory", "tags": []},
        {"query": "hello", "source": "memory", "tags": None},
        {"query": "hello", "source": "memory", "tags": "infra"},
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

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post(
        "/search",
        json={
            "query": "hello",
            "source": "memory",
            "kind": "decision",
            "tags": [" Infra ", "DATABASE", "infra"],
        },
    )
    assert response.status_code == 200
    assert captured["kind"] == "decision"
    assert captured["tags"] == [" Infra ", "DATABASE", "infra"]


def test_search_silently_ignores_include_atoms(monkeypatch):
    """No compat shim: a caller still sending include_atoms is accepted and ignored."""
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post(
        "/search",
        json={"query": "hello", "source": "memory", "include_atoms": False},
    )
    assert response.status_code == 200
    assert "include_atoms" not in captured


def test_search_forwards_raw_repo_filter(monkeypatch):
    """The route forwards the raw repo list as-is; search() does the normalizing."""
    captured = {}

    async def fake_search(query, source="all", **options):
        captured.update(options)
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post(
        "/search", json={"query": "marker", "source": "code", "repo": [" repo_a ", "repo_a"]}
    )
    assert response.status_code == 200
    assert captured["repo"] == [" repo_a ", "repo_a"]


def test_search_rejects_repo_filter_outside_code_source():
    response = client.post("/search", json={"query": "marker", "repo": ["repo_a"]})
    assert response.status_code == 400
    assert 'source="code"' in response.json()["error"]


def test_search_forwards_namespaces_filter(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(api, "search", fake_search)
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

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert "namespaces" not in captured


def test_search_malformed_namespaces_400():
    response = client.post("/search", json={"query": "hello", "namespaces": "team-a"})
    assert response.status_code == 400
    assert "namespaces" in response.json()["error"]


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


def test_search_forwards_min_score(monkeypatch):
    captured = {}

    async def fake_search(query, source="all", **options):
        captured.update(options)
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello", "min_score": 0.3})
    assert response.status_code == 200
    assert captured["min_score"] == 0.3


def test_search_omitted_min_score_does_not_reach_search(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(api, "search", fake_search)
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

    monkeypatch.setattr(api, "search", fake_search)
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


@pytest.mark.parametrize("tags", ["infra", {"tag": "infra"}, [1], ["infra", None]])
def test_save_memory_malformed_tags_400(tags):
    response = client.post("/save_memory", json={"content": "valid content", "tags": tags})
    assert response.status_code == 400
    assert response.json()["error"] == "tags must be a list of strings"


def test_save_memory_valid_content_delegates_to_save_note(monkeypatch):
    captured = {}

    async def fake_save_note(content, kind="note", tags=None, supersedes=None, namespace="default"):
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


def test_save_memory_omitted_namespace_defaults_to_default(monkeypatch):
    captured = {}

    async def fake_save_note(content, kind="note", tags=None, supersedes=None, namespace="default"):
        captured["namespace"] = namespace
        return {"id": "note:x", "kind": kind, "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post("/save_memory", json={"content": "distilled note text"})
    assert response.status_code == 200
    assert captured["namespace"] == "default"


def test_save_memory_forwards_explicit_namespace(monkeypatch):
    captured = {}

    async def fake_save_note(content, kind="note", tags=None, supersedes=None, namespace="default"):
        captured["namespace"] = namespace
        return {"id": "note:x", "kind": kind, "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory", json={"content": "distilled note text", "namespace": "team-a"}
    )
    assert response.status_code == 200
    assert captured["namespace"] == "team-a"


def test_save_memory_unregistered_namespace_400(monkeypatch):
    async def fake_save_note(content, kind="note", tags=None, supersedes=None, namespace="default"):
        raise ValueError(f"unregistered namespace: {namespace}")

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory", json={"content": "distilled note text", "namespace": "ghost"}
    )
    assert response.status_code == 400
    assert "unregistered namespace" in response.json()["error"]


@pytest.mark.parametrize("namespace", ["", "   ", 123, [], None])
def test_save_memory_malformed_namespace_400(namespace):
    response = client.post(
        "/save_memory", json={"content": "distilled note text", "namespace": namespace}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "namespace must be a non-empty string"


# ---- source rename: history rejected, memory accepted ----------------------


def test_search_history_source_rejected_with_400():
    response = client.post("/search", json={"query": "hello", "source": "history"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert "code" in error and "memory" in error and "all" in error


def test_search_memory_source_accepted(monkeypatch):
    async def fake_search(query, **options):
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello", "source": "memory"})
    assert response.status_code == 200


# ---- POST /search since/until ----------------------------------------------


def test_search_forwards_raw_since_and_until(monkeypatch):
    """The route forwards raw body values as-is; search() does the parsing."""
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post(
        "/search",
        json={"query": "q", "source": "memory", "since": "2026-08-01", "until": "2026-08-12"},
    )
    assert response.status_code == 200
    assert captured["since"] == "2026-08-01"
    assert captured["until"] == "2026-08-12"


def test_search_omitted_time_bounds_do_not_reach_search(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "q", "source": "memory"})
    assert response.status_code == 200
    assert "since" not in captured
    assert "until" not in captured


@pytest.mark.parametrize("field", ["since", "until"])
def test_search_explicit_null_time_bound_400(field):
    response = client.post("/search", json={"query": "q", "source": "memory", field: None})
    assert response.status_code == 400
    assert field in response.json()["error"]


@pytest.mark.parametrize(
    "body",
    [
        {"query": "q", "source": "memory", "since": "not-a-date"},
        {"query": "q", "source": "memory", "since": "2026-08-13", "until": "2026-08-12"},
        {"query": "q", "source": "code", "since": "2026-08-01"},
        {"query": "q", "source": "all", "until": "2026-08-01"},
    ],
)
def test_search_time_bound_validation_errors_are_400(body):
    response = client.post("/search", json=body)
    assert response.status_code == 400
    assert set(response.json()) == {"error"}
