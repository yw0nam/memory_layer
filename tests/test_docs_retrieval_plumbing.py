"""Unit contracts for retrieval filters and atom-parent resolution."""

from __future__ import annotations

import asyncio
import json

import pytest

from memory_base.retrieval.search import (
    PER_FILE_CAP,
    Hit,
    _dedup_cap,
    _merge_atom_hits,
    _search_atoms,
    _search_code,
    _search_memory,
    validate_search_options,
)


class FakeSearchConnection:
    def __init__(self, fetch_results):
        self.fetch_results = list(fetch_results)
        self.queries = []

    async def fetchval(self, query, *args):
        self.queries.append((query, args))
        return True

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.fetch_results.pop(0)


@pytest.mark.parametrize("kind", ["doc", "note", "decision"])
def test_memory_kind_filters_are_valid(kind):
    assert validate_search_options("memory", kind, None) == (kind, None, None)


@pytest.mark.parametrize("kind", ["atom", "code", "", 1])
def test_unknown_kind_filter_is_rejected(kind):
    with pytest.raises(ValueError, match="kind must be one of"):
        validate_search_options("memory", kind, None)


@pytest.mark.parametrize(
    ("source", "kind", "tags"),
    [
        ("all", "note", None),
        ("code", "decision", None),
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
    ) == (None, ["infrastructure", "database"], None)


@pytest.mark.parametrize("tags", [[], [" "], "infra", [1], ["infra", None]])
def test_empty_or_malformed_search_tags_are_rejected(tags):
    with pytest.raises(ValueError, match="tags"):
        validate_search_options("memory", None, tags)


def test_repo_filter_is_normalized_for_code_source():
    assert validate_search_options("code", None, None, repo=[" YUI ", "agent-team", "YUI"]) == (
        None,
        None,
        ["YUI", "agent-team"],
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


def test_memory_filters_are_inside_both_candidate_queries():
    conn = FakeSearchConnection([[], []])
    asyncio.run(
        _search_memory(
            conn,
            "query",
            "[1]",
            kind="decision",
            tags=["infra", "database"],
        )
    )

    vector_sql, vector_args = conn.queries[1]
    fts_sql, fts_args = conn.queries[2]
    for sql in (vector_sql, fts_sql):
        assert "chunk_kind <> 'atom'" in sql
        assert "chunk_kind =" in sql
        assert "metadata->'tags' ?|" in sql
        assert sql.index("chunk_kind =") < sql.index("LIMIT")
        assert sql.index("metadata->'tags' ?|") < sql.index("LIMIT")
    assert vector_args == ("[1]", "decision", ["infra", "database"])
    assert fts_args == ("query", "decision", ["infra", "database"])


def test_memory_hit_uses_search_ref_and_keeps_source_ref_for_dedup():
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
        "idf_score": None,
        "archived_at": None,
    }
    conn = FakeSearchConnection([[row], []])
    hits = asyncio.run(_search_memory(conn, "query", "[1]"))

    assert hits[0].ref == "guide.md#chunk-2"
    assert hits[0].meta["source_ref"] == "guide.md"
    assert hits[0].meta["kind"] == "doc"
    assert hits[0].meta["tags"] == ["infra"]


def test_csv_card_hit_uses_search_ref():
    row = {
        "id": "doc:table.csv:card:0",
        "source_ref": "table.csv",
        "chunk_kind": "doc",
        "metadata": {"search_ref": "table.csv#card-0", "tags": ["data"]},
        "distilled": "summary card",
        "content_raw": "summary card",
        "ts_last_active": 100.0,
        "idf_score": None,
        "archived_at": None,
    }
    conn = FakeSearchConnection([[row], []])
    hits = asyncio.run(_search_memory(conn, "query", "[1]"))
    assert hits[0].ref == "table.csv#card-0"


def test_memory_hit_falls_back_to_source_ref():
    row = {
        "id": "note:1",
        "source_ref": "save_memory",
        "chunk_kind": "note",
        "metadata": {},
        "distilled": "note",
        "content_raw": "note",
        "ts_last_active": 100.0,
        "idf_score": None,
        "archived_at": None,
    }
    conn = FakeSearchConnection([[row], []])
    hits = asyncio.run(_search_memory(conn, "query", "[1]"))
    assert hits[0].ref == "save_memory"


def test_document_chunks_remain_subject_to_per_file_cap():
    hits = [
        Hit(
            source="memory",
            ref=f"guide.md#chunk-{i}",
            text="",
            ts=0.0,
            rrf=10.0 - i,
            meta={"source_ref": "guide.md"},
        )
        for i in range(PER_FILE_CAP + 2)
    ]
    assert len(_dedup_cap(hits)) == PER_FILE_CAP


def test_atom_lane_collapses_parents_at_highest_cosine_and_skips_dangling_rows():
    rows = [
        {
            "atom_id": "parent:1:atom:0",
            "matched_question": "lower match",
            "atom_cosine": 0.72,
            "id": "parent:1",
            "source_ref": "guide.md",
            "chunk_kind": "doc",
            "content_raw": "parent text",
            "distilled": None,
            "ts_last_active": 100.0,
            "metadata": {"search_ref": "guide.md#chunk-1", "tags": ["infra"]},
            "archived_at": None,
        },
        {
            "atom_id": "parent:1:atom:1",
            "matched_question": "higher match",
            "atom_cosine": 0.91,
            "id": "parent:1",
            "source_ref": "guide.md",
            "chunk_kind": "doc",
            "content_raw": "parent text",
            "distilled": None,
            "ts_last_active": 100.0,
            "metadata": {"search_ref": "guide.md#chunk-1", "tags": ["infra"]},
            "archived_at": None,
        },
        {
            "atom_id": "parent:2:atom:0",
            "matched_question": "other parent",
            "atom_cosine": 0.80,
            "id": "parent:2",
            "source_ref": "other.md",
            "chunk_kind": "doc",
            "content_raw": "other text",
            "distilled": None,
            "ts_last_active": 90.0,
            "metadata": {"search_ref": "other.md#chunk-0", "tags": []},
            "archived_at": None,
        },
    ]
    conn = FakeSearchConnection([rows])
    hits = asyncio.run(_search_atoms(conn, "[1]"))

    assert [hit.meta["id"] for hit in hits] == ["parent:1", "parent:2"]
    assert hits[0].meta["atom_id"] == "parent:1:atom:1"
    assert hits[0].meta["atom_question"] == "higher match"
    assert hits[0].rrf == 0.0
    sql = conn.queries[0][0]
    assert "JOIN" in sql
    assert "parent.id = atom.metadata->>'parent_id'" in sql
    assert "parent.chunk_kind <> 'atom'" in sql
    assert "LIMIT" in sql


def test_atom_parent_filters_are_applied_before_the_candidate_limit():
    conn = FakeSearchConnection([[]])
    asyncio.run(
        _search_atoms(
            conn,
            "[1]",
            kind="decision",
            tags=["infra"],
        )
    )
    sql, args = conn.queries[0]
    assert sql.index("parent.chunk_kind =") < sql.index("ORDER BY")
    assert sql.index("parent.metadata->'tags' ?|") < sql.index("ORDER BY")
    assert "parent.archived_at IS NULL" in sql
    assert args == ("[1]", "decision", ["infra"])


def test_atom_evidence_annotates_baseline_parent_without_duplication():
    baseline = [
        Hit(
            source="memory",
            ref="guide.md#chunk-0",
            text="parent",
            ts=100.0,
            rrf=0.5,
            meta={"id": "parent:1", "source_ref": "guide.md"},
        )
    ]
    atom = Hit(
        source="memory",
        ref="guide.md#chunk-0",
        text="parent",
        ts=100.0,
        meta={
            "id": "parent:1",
            "atom_id": "parent:1:atom:0",
            "atom_question": "matched question",
            "atom_cosine": 0.9,
        },
    )
    merged = _merge_atom_hits(baseline, [atom])
    assert merged == baseline
    assert merged[0].meta["atom_id"] == "parent:1:atom:0"


def test_atom_union_has_no_shared_fused_top_truncation():
    baseline = [
        Hit(
            source="memory",
            ref=f"base:{index}",
            text="",
            ts=0.0,
            meta={"id": f"base:{index}"},
        )
        for index in range(20)
    ]
    atoms = [
        Hit(
            source="memory",
            ref=f"atom-parent:{index}",
            text="",
            ts=0.0,
            meta={
                "id": f"atom-parent:{index}",
                "atom_id": f"atom:{index}",
                "atom_question": "question",
                "atom_cosine": 1.0 - index / 100,
            },
        )
        for index in range(8)
    ]
    assert len(_merge_atom_hits(baseline, atoms)) == 28
