"""Unit contracts for retrieval filters."""

from __future__ import annotations

import asyncio
import inspect
import json

import asyncpg
import pytest

from memory_base.core.config import PG_SCHEMA
from memory_base.retrieval import search as search_module
from memory_base.retrieval.search import (
    PER_FILE_CAP,
    Hit,
    _dedup_cap,
    _search_code,
    _search_memory,
    history_predicates,
    normalize_namespaces,
    search,
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


def test_search_code_returns_no_hits_when_code_chunks_table_is_missing():
    conn = UndefinedTableConnection()
    hits = asyncio.run(_search_code(conn, "query", "[1]"))
    assert hits == []


def test_search_memory_returns_no_hits_when_memory_chunks_table_is_missing():
    conn = UndefinedTableConnection()
    hits = asyncio.run(_search_memory(conn, "query", "[1]"))
    assert hits == []


@pytest.mark.parametrize("kind", ["doc", "personal", "work"])
def test_memory_kind_filters_are_valid(kind):
    assert validate_search_options("memory", kind, None) == (kind, None, None, None, None)


@pytest.mark.parametrize("kind", ["code", "note", "decision", "", 1])
def test_unknown_kind_filter_is_rejected(kind):
    with pytest.raises(ValueError, match="kind must be one of"):
        validate_search_options("memory", kind, None)


@pytest.mark.parametrize(
    ("source", "kind", "tags"),
    [
        ("all", "personal", None),
        ("code", "work", None),
        ("all", None, ["infra"]),
        ("code", None, ["infra"]),
    ],
)
def test_filters_require_memory_source(source, kind, tags):
    with pytest.raises(ValueError, match='source="memory"'):
        validate_search_options(source, kind, tags)


def test_search_tags_are_normalized_with_any_semantics():
    assert validate_search_options(
        "memory", None, [" Infrastructure ", "DATABASE", "infrastructure"]
    ) == (None, ["infrastructure", "database"], None, None, None)


@pytest.mark.parametrize("tags", [[], [" "], "infra", [1], ["infra", None]])
def test_empty_or_malformed_search_tags_are_rejected(tags):
    with pytest.raises(ValueError, match="tags"):
        validate_search_options("memory", None, tags)


def test_repo_filter_is_normalized_for_code_source():
    assert validate_search_options("code", None, None, repo=[" YUI ", "agent-team", "YUI"]) == (
        None,
        None,
        ["YUI", "agent-team"],
        None,
        None,
    )


@pytest.mark.parametrize("source", ["all", "memory"])
def test_repo_filter_requires_code_source(source):
    with pytest.raises(ValueError, match='source="code"'):
        validate_search_options(source, None, None, repo=["YUI"])


@pytest.mark.parametrize("repo", [[], [" "], "YUI", [1], ["YUI", None]])
def test_empty_or_malformed_repo_filter_is_rejected(repo):
    with pytest.raises(ValueError, match="repo"):
        validate_search_options("code", None, None, repo=repo)


def test_repo_filter_is_inside_both_code_candidate_queries():
    conn = FakeSearchConnection([[], []])
    asyncio.run(_search_code(conn, "query", "[1]", repo=["YUI"]))
    code_queries = [(q, args) for q, args in conn.queries if "code_chunks" in q]
    assert len(code_queries) == 2
    assert all("repo = ANY($2::text[])" in q for q, _ in code_queries)
    assert all(args[-1] == ["YUI"] for _, args in code_queries)


def test_code_hit_carries_repo():
    row = {
        "id": 1,
        "repo": "YUI",
        "filename": "YUI/app/main.py",
        "code": "def main(): ...",
        "start_line": 1,
        "end_line": 2,
        "mtime": 100.0,
    }
    conn = FakeSearchConnection([[row], []])
    hits = asyncio.run(_search_code(conn, "query", "[1]"))
    assert hits[0].meta["repo"] == "YUI"
    assert all("repo = ANY" not in q for q, _ in conn.queries)


def test_search_code_fts_leg_qualifies_bm25_index_with_default_schema():
    conn = FakeSearchConnection([[], []])
    asyncio.run(_search_code(conn, "query", "[1]"))
    fts_sql, _ = conn.queries[1]
    assert fts_sql.count(f"to_bm25query($1, '{PG_SCHEMA}.code_chunks_bm25')") == 2


def test_search_code_fts_leg_qualifies_bm25_index_with_explicit_schema():
    conn = FakeSearchConnection([[], []])
    asyncio.run(_search_code(conn, "query", "[1]", schema="memory_eval_scratch"))
    fts_sql, _ = conn.queries[1]
    assert fts_sql.count("to_bm25query($1, 'memory_eval_scratch.code_chunks_bm25')") == 2
    assert f"'{PG_SCHEMA}.code_chunks_bm25'" not in fts_sql


# ---- namespaces filter -----------------------------------------------------


def test_normalize_namespaces_none_means_no_filter():
    assert normalize_namespaces(None) is None


def test_normalize_namespaces_empty_list_means_no_filter():
    assert normalize_namespaces([]) is None


def test_normalize_namespaces_dedupes_and_strips():
    assert normalize_namespaces([" team-a ", "team-b", "team-a"]) == ["team-a", "team-b"]


@pytest.mark.parametrize("namespaces", ["team-a", [1], [None]])
def test_normalize_namespaces_rejects_non_list_of_str(namespaces):
    with pytest.raises(ValueError, match="namespaces"):
        normalize_namespaces(namespaces)


def test_history_predicates_omits_namespace_clause_by_default():
    predicates, args = history_predicates(include_archived=False, kind=None, tags=None)
    assert "namespace" not in predicates
    assert "atom" not in predicates
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


def test_search_memory_fts_leg_qualifies_bm25_index_with_default_schema():
    conn = FakeSearchConnection([[], []])
    asyncio.run(_search_memory(conn, "query", "[1]"))
    fts_sql, _ = conn.queries[1]
    assert fts_sql.count(f"to_bm25query($1, '{PG_SCHEMA}.memory_chunks_bm25')") == 2


def test_search_memory_fts_leg_qualifies_bm25_index_with_explicit_schema():
    conn = FakeSearchConnection([[], []])
    asyncio.run(_search_memory(conn, "query", "[1]", schema="memory_eval_scratch"))
    fts_sql, _ = conn.queries[1]
    assert fts_sql.count("to_bm25query($1, 'memory_eval_scratch.memory_chunks_bm25')") == 2
    assert f"'{PG_SCHEMA}.memory_chunks_bm25'" not in fts_sql


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


def test_memory_hit_uses_search_ref_and_carries_cap_fields():
    metadata = {
        "search_ref": "guide.md#chunk-2",
        "tags": ["infra"],
    }
    row = {
        "id": "doc:guide.md:2",
        "source_ref": "guide.md",
        "chunk_kind": "doc",
        "metadata": json.dumps(metadata),
        "distilled": None,
        "content_raw": "document chunk",
        "ts_last_active": 100.0,
        "archived_at": None,
        "namespace": "default",
        "session_id": "guide.md",
        "occurred_at": None,
    }
    conn = FakeSearchConnection([[row], []])
    hits = asyncio.run(_search_memory(conn, "query", "[1]"))

    assert hits[0].ref == "guide.md#chunk-2"
    assert hits[0].meta["namespace"] == "default"
    assert hits[0].meta["session_id"] == "guide.md"
    assert hits[0].meta["kind"] == "doc"
    assert hits[0].meta["tags"] == ["infra"]
    assert "source_ref" not in hits[0].meta


def test_csv_card_hit_uses_search_ref():
    row = {
        "id": "doc:table.csv:card:0",
        "source_ref": "table.csv",
        "chunk_kind": "doc",
        "metadata": {
            "search_ref": "table.csv#card-0",
            "tags": ["data"],
            "columns": ["group", "value"],
        },
        "distilled": "summary card",
        "content_raw": "summary card",
        "ts_last_active": 100.0,
        "archived_at": None,
        "namespace": "default",
        "session_id": "table.csv",
        "occurred_at": None,
    }
    conn = FakeSearchConnection([[row], []])
    hits = asyncio.run(_search_memory(conn, "query", "[1]"))
    assert hits[0].ref == "table.csv#card-0"
    assert hits[0].meta["columns"] == ["group", "value"]


def test_memory_hit_falls_back_to_source_ref():
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


def test_author_filter_is_accepted_for_memory_source():
    assert validate_search_options("memory", None, None, author="natsume") == (
        None,
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize("source", ["all", "code"])
def test_author_filter_requires_memory_source(source):
    with pytest.raises(ValueError, match='author filter requires source="memory"'):
        validate_search_options(source, None, None, author="natsume")


@pytest.mark.parametrize("author", [1, ["natsume"], {"a": 1}])
def test_malformed_author_filter_is_rejected(author):
    with pytest.raises(ValueError, match="author"):
        validate_search_options("memory", None, None, author=author)


def test_history_predicates_adds_author_clause():
    predicates, args = history_predicates(
        include_archived=True, kind=None, tags=None, author="natsume"
    )
    assert predicates == "metadata->>'author' = $2"
    assert args == ["natsume"]


def test_history_predicates_author_clause_numbers_last():
    predicates, args = history_predicates(
        include_archived=False,
        kind="work",
        tags=["infra"],
        namespaces=["team-a"],
        since=1.0,
        until=2.0,
        author="natsume",
    )
    assert args == ["work", ["infra"], ["team-a"], 1.0, 2.0, "natsume"]
    assert "metadata->>'author' = $7" in predicates


def test_search_memory_filters_by_author():
    conn = FakeSearchConnection([[], []])
    asyncio.run(_search_memory(conn, "query", "[1]", author="natsume"))
    for sql, args in conn.queries:
        assert "metadata->>'author' = $" in sql
        assert args[-1] == "natsume"


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


# ---- _dedup_cap keys memory hits on (namespace, session_id) ---------------


def _memory_hit(ref, rrf, namespace, session_id):
    return Hit(
        source="memory",
        ref=ref,
        text="",
        ts=0.0,
        rrf=rrf,
        meta={"namespace": namespace, "session_id": session_id},
    )


def test_notes_with_distinct_session_ids_are_not_capped():
    # all save_memory notes share one ref; only session_id tells them apart.
    hits = [
        _memory_hit("save_memory", rrf=10.0 - i, namespace="default", session_id=f"note:{i}")
        for i in range(5)
    ]
    assert len(_dedup_cap(hits)) == 5


def test_document_chunks_sharing_a_session_id_remain_capped():
    hits = [
        _memory_hit(f"guide.md#chunk-{i}", rrf=10.0 - i, namespace="default", session_id="guide.md")
        for i in range(PER_FILE_CAP + 2)
    ]
    assert len(_dedup_cap(hits)) == PER_FILE_CAP


def test_same_session_id_in_different_namespaces_is_not_shared():
    hits = [
        _memory_hit(f"doc#chunk-{i}", rrf=10.0 - i, namespace="team-a", session_id="doc")
        for i in range(PER_FILE_CAP + 2)
    ] + [
        _memory_hit(f"doc#chunk-{i}", rrf=5.0 - i, namespace="team-b", session_id="doc")
        for i in range(PER_FILE_CAP + 2)
    ]
    out = _dedup_cap(hits)
    assert len(out) == PER_FILE_CAP * 2
    assert sum(1 for h in out if h.meta["namespace"] == "team-a") == PER_FILE_CAP
    assert sum(1 for h in out if h.meta["namespace"] == "team-b") == PER_FILE_CAP


def test_search_does_not_accept_include_atoms():
    assert "include_atoms" not in inspect.signature(search).parameters


def test_atom_retrieve_k_constant_is_gone():
    assert not hasattr(search_module, "ATOM_RETRIEVE_K")
