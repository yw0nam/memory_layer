"""Unit tests for the REST API server (memory_base.serve.api).

No DB, no network: routes only parse/validate/delegate/JSON, so the hooks a route calls into are monkeypatched
at the module level, matching the existing convention (e.g.
``monkeypatch.setattr(answer, "search", fake_search)`` in test_answer_mcp.py):

- ``api.search``       -- delegate for POST /search (memory_base.retrieval.search.search)
- ``api.save_note``    -- delegate for POST /save_memory (memory_base.serve.notes.save_note)
- ``api.log_retrieval``-- best-effort access logging after a search (memory_base.serve.access_log)
- ``api.db_healthy``, ``api.embedding_healthy``, ``api.rerank_healthy``, ``api.llm_healthy``
  -- async dependency probes used by GET /health
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
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "checks": {"db": True, "embedding": True, "rerank": True, "llm": True},
    }


def test_health_503_when_db_unreachable(_all_probes_up, monkeypatch):
    async def fake_db_healthy():
        raise ConnectionError("db down")

    monkeypatch.setattr(api, "db_healthy", fake_db_healthy)
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["db"] is False


def test_health_503_when_embedding_unreachable(_all_probes_up, monkeypatch):
    async def fake_embedding_healthy():
        return False

    monkeypatch.setattr(api, "embedding_healthy", fake_embedding_healthy)
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["embedding"] is False


def test_health_503_when_rerank_unreachable(_all_probes_up, monkeypatch):
    async def fake_rerank_healthy():
        return False

    monkeypatch.setattr(api, "rerank_healthy", fake_rerank_healthy)
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert body["checks"]["rerank"] is False


def test_health_200_when_only_llm_unreachable(_all_probes_up, monkeypatch):
    async def fake_llm_healthy():
        return False

    monkeypatch.setattr(api, "llm_healthy", fake_llm_healthy)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"db": True, "embedding": True, "rerank": True, "llm": False}


def test_health_probe_raising_is_reported_as_false_not_propagated(_all_probes_up, monkeypatch):
    async def fake_llm_healthy():
        raise TimeoutError("connect timed out")

    monkeypatch.setattr(api, "llm_healthy", fake_llm_healthy)
    response = client.get("/health")
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
        {"query": "hello", "source": "memory", "kind": "atom"},
        {"query": "hello", "source": "memory", "tags": []},
        {"query": "hello", "source": "memory", "tags": None},
        {"query": "hello", "source": "memory", "tags": "infra"},
        {"query": "hello", "include_atoms": "yes"},
    ],
)
def test_search_filter_validation_errors_are_400(body):
    response = client.post("/search", json=body)
    assert response.status_code == 400
    assert set(response.json()) == {"error"}


def test_search_forwards_normalized_filters_and_include_atoms(monkeypatch):
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
            "include_atoms": False,
        },
    )
    assert response.status_code == 200
    assert captured["kind"] == "decision"
    assert captured["tags"] == ["infra", "database"]
    assert captured["include_atoms"] is False


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


@pytest.mark.parametrize("tags", ["infra", {"tag": "infra"}, [1], ["infra", None]])
def test_save_memory_malformed_tags_400(tags):
    response = client.post("/save_memory", json={"content": "valid content", "tags": tags})
    assert response.status_code == 400
    assert response.json()["error"] == "tags must be a list of strings"


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


# ---- POST /search/deep ------------------------------------------------------


def test_deep_search_missing_query_400():
    response = client.post("/search/deep", json={})
    assert response.status_code == 400
    assert "error" in response.json()


def test_deep_search_empty_query_400():
    response = client.post("/search/deep", json={"query": "   "})
    assert response.status_code == 400
    assert "error" in response.json()


def test_deep_search_invalid_max_hops_400(monkeypatch):
    async def fake_deep_search(query, **kwargs):
        raise ValueError("max_hops must be between 1 and 3")

    monkeypatch.setattr(api, "deep_search", fake_deep_search)
    response = client.post("/search/deep", json={"query": "hello", "max_hops": 0})
    assert response.status_code == 400
    assert "max_hops" in response.json()["error"]


def test_deep_search_invalid_kind_400(monkeypatch):
    async def fake_deep_search(query, **kwargs):
        raise ValueError("kind must be one of ('doc', 'note', 'decision')")

    monkeypatch.setattr(api, "deep_search", fake_deep_search)
    response = client.post("/search/deep", json={"query": "hello", "kind": "atom"})
    assert response.status_code == 400
    assert "kind" in response.json()["error"]


def test_deep_search_serialization_shape(monkeypatch):
    from memory_base.retrieval.decompose import DeepResult, EvidenceEntry, TraceEntry

    evidence = [
        EvidenceEntry(
            id="doc:guide.md:0",
            ref="guide.md#chunk-0",
            text="x" * 3000,
            kind="doc",
            tags=["infra"],
            date=1_700_000_000.0,
            hop=1,
            atom_question="what is the guide about?",
        ),
        EvidenceEntry(
            id="note:abc",
            ref="save_memory",
            text="short text",
            kind="note",
            tags=[],
            date=1_700_000_000.0,
            hop=2,
            atom_question=None,
        ),
    ]
    trace = [
        TraceEntry(hop=1, sub_questions=["sq1"], selected_ref="guide.md#chunk-0"),
        TraceEntry(hop=2, sub_questions=["sq2"], selected_ref="save_memory"),
    ]
    result = DeepResult(evidence=evidence, trace=trace, hops_used=2, stopped_reason="max_hops")

    async def fake_deep_search(query, **kwargs):
        return result

    monkeypatch.setattr(api, "deep_search", fake_deep_search)
    response = client.post("/search/deep", json={"query": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["hops_used"] == 2
    assert body["stopped_reason"] == "max_hops"
    assert len(body["evidence"]) == 2
    assert len(body["trace"]) == 2
    ev0 = body["evidence"][0]
    assert ev0["ref"] == "guide.md#chunk-0"
    assert ev0["kind"] == "doc"
    assert ev0["tags"] == ["infra"]
    assert ev0["hop"] == 1
    assert ev0["atom_question"] == "what is the guide about?"
    assert ev0["id"] == "doc:guide.md:0"
    assert ev0["date"] == "2023-11-14"
    assert len(ev0["text"]) == 2000
    ev1 = body["evidence"][1]
    assert ev1["atom_question"] is None
    assert ev1["text"] == "short text"
    tr0 = body["trace"][0]
    assert tr0["hop"] == 1
    assert tr0["sub_questions"] == ["sq1"]
    assert tr0["selected_ref"] == "guide.md#chunk-0"


def test_deep_search_forwards_options(monkeypatch):
    captured = {}

    async def fake_deep_search(query, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        from memory_base.retrieval.decompose import DeepResult

        return DeepResult(evidence=[], trace=[], hops_used=0, stopped_reason="done")

    monkeypatch.setattr(api, "deep_search", fake_deep_search)
    response = client.post(
        "/search/deep",
        json={
            "query": "multi-hop question",
            "max_hops": 2,
            "kind": "decision",
            "tags": ["infra"],
            "include_archived": True,
        },
    )
    assert response.status_code == 200
    assert captured["query"] == "multi-hop question"
    assert captured["max_hops"] == 2
    assert captured["kind"] == "decision"
    assert captured["tags"] == ["infra"]
    assert captured["include_archived"] is True


def test_deep_search_malformed_json_400():
    response = client.post(
        "/search/deep",
        content=b"{not valid json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert "error" in response.json()
