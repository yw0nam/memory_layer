"""Unit contracts for note retrieval filters."""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from memory_base.retrieval import search as search_module
from memory_base.retrieval.search import (
    _search_memory,
    history_predicates,
    normalize_namespaces,
    validate_search_options,
)


class FakeSearchConnection:
    def __init__(self, fetch_results):
        self.fetch_results = list(fetch_results)
        self.queries = []

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.fetch_results.pop(0)


class UndefinedTableConnection:
    async def fetch(self, query, *args):
        raise asyncpg.exceptions.UndefinedTableError("relation does not exist")


def test_search_memory_returns_no_hits_when_memory_chunks_table_is_missing():
    conn = UndefinedTableConnection()
    hits = asyncio.run(_search_memory(conn, "query", "[1]"))
    assert hits == []


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_memory_kind_filters_are_valid(kind):
    assert validate_search_options(kind, None) == (kind, None, None, None)


def test_unknown_kind_filter_is_rejected():
    with pytest.raises(ValueError, match="kind must be one of"):
        validate_search_options("note", None)


def test_search_tags_are_normalized_with_any_semantics():
    assert validate_search_options(None, [" Infrastructure ", "DATABASE", "infrastructure"]) == (
        None,
        ["infrastructure", "database"],
        None,
        None,
    )


@pytest.mark.parametrize("tags", [[], [" "], "infra", [1], ["infra", None]])
def test_empty_or_malformed_search_tags_are_rejected(tags):
    with pytest.raises(ValueError, match="tags"):
        validate_search_options(None, tags)


# ---- namespaces filter -----------------------------------------------------


def test_normalize_namespaces_none_means_no_filter():
    assert normalize_namespaces(None) is None


def test_normalize_namespaces_empty_list_means_no_filter():
    assert normalize_namespaces([]) is None


def test_normalize_namespaces_dedupes_and_strips():
    assert normalize_namespaces([" team-a ", "team-b", "team-a"]) == ["team-a", "team-b"]


@pytest.mark.parametrize("namespaces", ["team-a", [1]])
def test_normalize_namespaces_rejects_non_list_of_str(namespaces):
    with pytest.raises(ValueError, match="namespaces"):
        normalize_namespaces(namespaces)


def test_history_predicates_omits_namespace_clause_by_default():
    predicates, args = history_predicates(include_archived=False, kind=None, tags=None)
    assert "namespace" not in predicates
    assert args == []


def test_history_predicates_without_filters_returns_true():
    predicates, args = history_predicates(include_archived=True, kind=None, tags=None)
    assert predicates == "true"
    assert args == []


def test_history_predicates_with_filters_does_not_seed_true():
    predicates, _ = history_predicates(include_archived=False, kind=None, tags=None)
    assert "true AND" not in predicates


def test_history_predicates_adds_namespace_any_clause():
    predicates, args = history_predicates(
        include_archived=False, kind=None, tags=None, namespaces=["team-a", "team-b"]
    )
    assert "namespace = ANY($2::text[])" in predicates
    assert args == [["team-a", "team-b"]]


def test_history_predicates_namespace_clause_uses_alias():
    predicates, _ = history_predicates(
        include_archived=False, kind=None, tags=None, alias="parent", namespaces=["team-a"]
    )
    assert "parent.namespace = ANY(" in predicates


def test_search_memory_filters_by_namespace():
    conn = FakeSearchConnection([[], []])
    asyncio.run(_search_memory(conn, "query", "[1]", namespaces=["team-a"]))
    vector_sql, vector_args = conn.queries[0]
    fts_sql, fts_args = conn.queries[1]
    for sql in (vector_sql, fts_sql):
        assert "namespace = ANY($" in sql
    assert vector_args == ("[1]", ["team-a"])
    assert fts_args == ("query", ["team-a"])


def test_memory_filters_are_inside_both_candidate_queries():
    conn = FakeSearchConnection([[], []])
    asyncio.run(
        _search_memory(
            conn,
            "query",
            "[1]",
            kind="work",
            tags=["infra", "database"],
        )
    )

    vector_sql, vector_args = conn.queries[0]
    fts_sql, fts_args = conn.queries[1]
    for sql in (vector_sql, fts_sql):
        assert "chunk_kind =" in sql
        assert "metadata->'tags' ?|" in sql
        assert sql.index("chunk_kind =") < sql.index("LIMIT")
        assert sql.index("metadata->'tags' ?|") < sql.index("LIMIT")
    assert vector_args == ("[1]", "work", ["infra", "database"])
    assert fts_args == ("query", "work", ["infra", "database"])


def test_memory_hit_uses_source_ref():
    row = {
        "id": "note:1",
        "source_ref": "save_memory",
        "chunk_kind": "work",
        "metadata": {},
        "distilled": "note",
        "content_raw": "note",
        "ts_last_active": 100.0,
        "archived_at": None,
        "namespace": "default",
        "session_id": "note:1",
        "occurred_at": None,
    }
    conn = FakeSearchConnection([[row], []])
    hits = asyncio.run(_search_memory(conn, "query", "[1]"))
    assert hits[0].ref == "save_memory"


def test_memory_fusion_uses_vector_fts_and_recency_voters(monkeypatch):
    def row(cid, ts):
        return {
            "id": cid,
            "source_ref": "save_memory",
            "chunk_kind": "work",
            "metadata": {},
            "distilled": cid,
            "content_raw": cid,
            "ts_last_active": ts,
            "archived_at": None,
            "namespace": "default",
            "session_id": cid,
            "occurred_at": None,
        }

    calls = []
    real_rrf_fuse = search_module.rrf_fuse

    def recording_rrf_fuse(lists, weights=None):
        calls.append((lists, weights))
        return real_rrf_fuse(lists, weights)

    monkeypatch.setattr(search_module, "rrf_fuse", recording_rrf_fuse)
    vec_rows = [row("vec-old", 100.0), row("both", 200.0)]
    fts_rows = [row("both", 200.0), row("fts-new", 300.0)]
    conn = FakeSearchConnection([vec_rows, fts_rows])

    asyncio.run(_search_memory(conn, "query", "[1]"))

    assert calls == [
        (
            [["vec-old", "both"], ["both", "fts-new"], ["fts-new", "both", "vec-old"]],
            [1.0, search_module.FTS_RRF_WEIGHT, search_module.TIEBREAK_RRF_WEIGHT],
        )
    ]


# ---- author filter ---------------------------------------------------------


def test_author_filter_is_accepted():
    assert validate_search_options(None, None, authors=["natsume"]) == (None, None, None, None)


def test_malformed_author_filter_is_rejected():
    with pytest.raises(ValueError, match="author"):
        validate_search_options(None, None, authors="natsume")


def test_history_predicates_adds_author_clause():
    predicates, args = history_predicates(
        include_archived=True, kind=None, tags=None, authors=["natsume"]
    )
    assert predicates == "metadata->>'author' = ANY($2::text[])"
    assert args == [["natsume"]]


def test_history_predicates_author_clause_numbers_last():
    predicates, args = history_predicates(
        include_archived=False,
        kind="work",
        tags=["infra"],
        namespaces=["team-a"],
        since=1.0,
        until=2.0,
        authors=["natsume"],
    )
    assert args == ["work", ["infra"], ["team-a"], 1.0, 2.0, ["natsume"]]
    assert "metadata->>'author' = ANY($7::text[])" in predicates


def test_search_memory_filters_by_author():
    conn = FakeSearchConnection([[], []])
    asyncio.run(_search_memory(conn, "query", "[1]", authors=["natsume"]))
    for sql, args in conn.queries:
        assert "metadata->>'author' = ANY($" in sql
        assert args[-1] == ["natsume"]


def test_memory_hit_carries_the_author():
    row = {
        "id": "note:1",
        "source_ref": "save_memory",
        "chunk_kind": "work",
        "metadata": {"author": "natsume"},
        "distilled": "note",
        "content_raw": "note",
        "ts_last_active": 100.0,
        "archived_at": None,
        "namespace": "default",
        "session_id": "note:1",
        "occurred_at": None,
    }
    conn = FakeSearchConnection([[row], []])
    hits = asyncio.run(_search_memory(conn, "query", "[1]"))
    assert hits[0].meta["author"] == "natsume"


def test_memory_hit_author_is_none_when_unrecorded():
    row = {
        "id": "note:1",
        "source_ref": "save_memory",
        "chunk_kind": "work",
        "metadata": {},
        "distilled": "note",
        "content_raw": "note",
        "ts_last_active": 100.0,
        "archived_at": None,
        "namespace": "default",
        "session_id": "note:1",
        "occurred_at": None,
    }
    conn = FakeSearchConnection([[row], []])
    hits = asyncio.run(_search_memory(conn, "query", "[1]"))
    assert hits[0].meta["author"] is None
