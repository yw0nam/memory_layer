"""Unit coverage for the reproducible retrieval evaluation harness."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from memory_base.adapters.document import chunk_markdown, read_csv_sample
from memory_base.eval import retrieval

FIXTURES = Path(__file__).parent / "fixtures"
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


def test_gate_applies_class_tolerance_and_requires_no_overall_drop():
    atoms_off = {
        "work": retrieval.MetricSummary(5, 0.8, 0.7),
        "kb": retrieval.MetricSummary(5, 0.6, 0.5),
        "overall": retrieval.MetricSummary(10, 0.7, 0.6),
    }
    passing = {
        "work": retrieval.MetricSummary(5, 0.75, 0.7),
        "kb": retrieval.MetricSummary(5, 0.65, 0.5),
        "overall": retrieval.MetricSummary(10, 0.7, 0.6),
    }
    overall_drop = dict(passing, overall=retrieval.MetricSummary(10, 0.69, 0.6))
    class_drop = dict(passing, work=retrieval.MetricSummary(5, 0.74, 0.7))
    assert retrieval.gate_passes(atoms_off, passing)
    assert not retrieval.gate_passes(atoms_off, overall_drop)
    assert not retrieval.gate_passes(atoms_off, class_drop)


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
    assert "Gate verdict: NOT RUN" in output


def test_path_completion_all_covered():
    labels = [
        retrieval.EvalLabel("q1", "multi_hop", ("doc:a.md:0", "doc:b.md:1")),
        retrieval.EvalLabel("q2", "multi_hop", ("doc:c.md:0", "doc:d.md:2")),
    ]
    evidence = [
        ["doc:a.md:0", "doc:b.md:1"],
        ["doc:c.md:0", "doc:d.md:2"],
    ]
    assert retrieval.path_completion(labels, evidence) == 1.0


def test_path_completion_partial():
    labels = [
        retrieval.EvalLabel("q1", "multi_hop", ("doc:a.md:0", "doc:b.md:1")),
        retrieval.EvalLabel("q2", "multi_hop", ("doc:c.md:0", "doc:d.md:2")),
    ]
    evidence = [
        ["doc:a.md:0", "doc:b.md:1"],
        ["doc:c.md:0"],
    ]
    assert retrieval.path_completion(labels, evidence) == 0.5


def test_path_completion_empty_evidence():
    labels = [
        retrieval.EvalLabel("q1", "multi_hop", ("doc:a.md:0", "doc:b.md:1")),
    ]
    evidence = [[]]
    assert retrieval.path_completion(labels, evidence) == 0.0


def test_path_completion_empty_labels():
    assert retrieval.path_completion([], []) == 0.0


def test_deep_gate_passes_with_sufficient_recall_and_completion():
    baseline = retrieval.MetricSummary(6, 0.5, 0.4)
    deep = retrieval.MetricSummary(6, 0.6, 0.5)
    assert retrieval.deep_gate_passes(baseline, deep, 0.5)


def test_deep_gate_fails_on_low_completion():
    baseline = retrieval.MetricSummary(6, 0.5, 0.4)
    deep = retrieval.MetricSummary(6, 0.6, 0.5)
    assert not retrieval.deep_gate_passes(baseline, deep, 0.3)


def test_deep_gate_fails_on_lower_recall():
    baseline = retrieval.MetricSummary(6, 0.5, 0.4)
    deep = retrieval.MetricSummary(6, 0.4, 0.3)
    assert not retrieval.deep_gate_passes(baseline, deep, 0.5)


def test_deep_gate_recall_tie_passes_only_with_completion():
    baseline = retrieval.MetricSummary(6, 0.5, 0.4)
    deep = retrieval.MetricSummary(6, 0.5, 0.4)
    assert retrieval.deep_gate_passes(baseline, deep, 0.4)
    assert not retrieval.deep_gate_passes(baseline, deep, 0.39)


def test_multi_hop_fixture_integrity():
    labels = retrieval.load_labels(FIXTURES / "retrieval_eval.jsonl")
    multi_hop = [label for label in labels if label.query_class == "multi_hop"]
    assert len(multi_hop) >= 5
    for label in multi_hop:
        docs = {retrieval._document_of(rid) for rid in label.relevant_ids}
        assert len(docs) >= 2
        assert all(retrieval.RELEVANT_ID_RE.fullmatch(rid) for rid in label.relevant_ids)


def test_single_hop_gate_excludes_multi_hop():
    labels = retrieval.load_labels(FIXTURES / "retrieval_eval.jsonl")
    single_hop = [label for label in labels if label.query_class != retrieval.MULTI_HOP_CLASS]
    single_classes = {label.query_class for label in single_hop}
    assert retrieval.MULTI_HOP_CLASS not in single_classes
    assert single_classes == retrieval.SINGLE_HOP_CLASSES


def test_baseline_search_kwargs_are_pinned():
    assert retrieval.BASELINE_SEARCH_KWARGS == {
        "source": "memory",
        "include_atoms": True,
        "rerank": True,
        "include_archived": False,
    }
