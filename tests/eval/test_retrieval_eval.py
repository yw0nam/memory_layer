"""Unit coverage for the reproducible retrieval evaluation harness."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import pytest

from memory_base.adapters.document import chunk_markdown, read_csv_sample
from memory_base.eval import retrieval

FIXTURES = Path(__file__).parents[1] / "fixtures"
EVAL_DOCS = FIXTURES / "eval_docs"


def test_recall_at_five_uses_all_labeled_relevant_rows():
    retrieved = ["unrelated-1", "relevant-a", "unrelated-2", "unrelated-3", "unrelated-4"]
    assert retrieval.recall_at_k(retrieved, {"relevant-a", "relevant-b"}, k=5) == 0.5
    assert (
        retrieval.recall_at_k(["relevant-a", "relevant-b"], {"relevant-a", "relevant-b"}, k=5)
        == 1.0
    )


def test_mrr_at_ten_uses_first_relevant_rank_and_cutoff():
    retrieved = [f"row-{index}" for index in range(1, 12)]
    assert retrieval.reciprocal_rank_at_k(retrieved, {"row-3", "row-7"}, k=10) == pytest.approx(
        1 / 3
    )
    assert retrieval.reciprocal_rank_at_k(retrieved, {"row-11"}, k=10) == 0.0


def test_metric_aggregation_reports_each_class_and_overall():
    results = [
        retrieval.QueryMetrics("work", 1.0, 1.0),
        retrieval.QueryMetrics("work", 0.0, 0.5),
        retrieval.QueryMetrics("kb", 0.5, 0.25),
    ]
    summary = retrieval.aggregate_metrics(results)
    assert summary["work"] == retrieval.MetricSummary(2, 0.5, 0.75)
    assert summary["kb"] == retrieval.MetricSummary(1, 0.5, 0.25)
    assert summary["overall"].count == 3
    assert summary["overall"].recall_at_5 == 0.5
    assert summary["overall"].mrr_at_10 == pytest.approx(7 / 12)


def test_missing_relevant_ids_remain_recall_misses():
    label = retrieval.EvalLabel(
        query="Find both rows",
        query_class="work",
        relevant_ids=("doc:guide.md:0", "doc:guide.md:1"),
    )
    result = retrieval.score_query(
        label,
        retrieved_ids=["doc:guide.md:0"],
        corpus_ids={"doc:guide.md:0"},
    )
    assert result.recall_at_5 == 0.5
    assert result.mrr_at_10 == 1.0
    assert result.missing_relevant_ids == ("doc:guide.md:1",)


def test_eval_file_integrity_and_class_coverage():
    labels = retrieval.load_labels(FIXTURES / "retrieval_eval.jsonl")
    counts = Counter(label.query_class for label in labels)
    assert 20 <= len(labels) <= 40
    assert set(counts) == {
        "agent_work_history",
        "companion_fragment",
        "personal_kb",
        "multi_hop",
    }
    assert all(count >= 5 for count in counts.values())
    assert all(label.relevant_ids for label in labels)
    assert all(
        retrieval.RELEVANT_ID_RE.fullmatch(row_id)
        for label in labels
        for row_id in label.relevant_ids
    )
    companion = [label.query for label in labels if label.query_class == "companion_fragment"]
    assert companion
    assert all("?" not in query for query in companion)


def test_labeled_relevant_ids_are_produced_by_fixture_chunking():
    produced_ids: set[str] = set()
    for path in sorted(EVAL_DOCS.iterdir()):
        if path.suffix == ".csv":
            read_csv_sample(path)
            produced_ids.add(f"doc:{path.name}:card:0")
            continue
        result = chunk_markdown(path.read_text(encoding="utf-8"))
        produced_ids.update(f"doc:{path.name}:{chunk.ordinal}" for chunk in result.chunks)

    labels = retrieval.load_labels(FIXTURES / "retrieval_eval.jsonl")
    labeled_ids = {row_id for label in labels for row_id in label.relevant_ids}
    assert labeled_ids <= produced_ids


def test_markdown_fixtures_include_varied_headings_sizes_and_multilingual_context():
    markdown_paths = sorted(EVAL_DOCS.glob("*.md"))
    assert len(markdown_paths) >= 3
    chunks = [
        chunk
        for path in markdown_paths
        for chunk in chunk_markdown(path.read_text(encoding="utf-8")).chunks
    ]
    assert {len(chunk.heading_path) for chunk in chunks} >= {1, 2}
    assert len({len(chunk.text) // 50 for chunk in chunks}) >= 3

    multilingual = chunk_markdown(
        (EVAL_DOCS / "multilingual-runbook.md").read_text(encoding="utf-8")
    ).chunks
    non_ascii = [not chunk.text.isascii() for chunk in multilingual]
    assert non_ascii == [False, True, False]


def test_main_reports_unavailable_prerequisites_without_raising(monkeypatch, capsys):
    async def unavailable():
        raise RuntimeError("service offline")

    monkeypatch.setattr(retrieval, "run_evaluation", unavailable)
    retrieval.main()
    output = capsys.readouterr().out
    assert "Evaluation unavailable: service offline" in output
    assert "Report NOT RUN" in output


def test_scratch_schema_scope_patches_and_restores_schema_module():
    """ensure_schema[_once] read schema_module.PG_SCHEMA unparameterized, so the eval
    still rebinds it for the run; search() and run_document_job() do not need this."""
    from memory_base.eval import retrieval as eval_module

    original = eval_module.schema_module.PG_SCHEMA
    with eval_module._scratch_schema_scope("scratch_test_schema"):
        assert eval_module.schema_module.PG_SCHEMA == "scratch_test_schema"
    assert eval_module.schema_module.PG_SCHEMA == original


def test_scratch_schema_scope_restores_on_exception():
    from memory_base.eval import retrieval as eval_module

    original = eval_module.schema_module.PG_SCHEMA
    with pytest.raises(RuntimeError, match="boom"):
        with eval_module._scratch_schema_scope("scratch_test_schema"):
            raise RuntimeError("boom")
    assert eval_module.schema_module.PG_SCHEMA == original


def test_ingest_fixture_threads_schema_to_run_document_job(monkeypatch, tmp_path):
    """ingest_api.PG_SCHEMA reads are contained in run_document_job's own call graph,
    so the eval passes schema explicitly instead of rebinding ingest_api's global."""
    from memory_base.eval import retrieval as eval_module

    captured = {}

    async def fake_run_document_job(job, upload_path, filename, mode, origin, schema=None):
        captured["schema"] = schema
        job.status = "succeeded"

    monkeypatch.setattr(eval_module.ingest_api, "run_document_job", fake_run_document_job)
    fixture = tmp_path / "note.md"
    fixture.write_text("hello", encoding="utf-8")

    asyncio.run(eval_module._ingest_fixture(fixture, "scratch_test_schema"))

    assert captured["schema"] == "scratch_test_schema"


def test_evaluate_mode_threads_schema_to_search(monkeypatch):
    """search_module.PG_SCHEMA reads are contained in search()'s own call graph, so
    the eval passes schema explicitly instead of rebinding search_module's global."""
    from memory_base.eval import retrieval as eval_module

    captured = {}

    async def fake_search(query, **kwargs):
        captured["schema"] = kwargs.get("schema")
        return []

    monkeypatch.setattr(eval_module.search_module, "search", fake_search)
    label = eval_module.EvalLabel(query="q", query_class="work", relevant_ids=("doc:x:0",))

    asyncio.run(eval_module._evaluate_mode([label], set(), "scratch_test_schema"))

    assert captured["schema"] == "scratch_test_schema"
