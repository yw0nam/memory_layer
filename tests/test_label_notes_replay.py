"""Unit coverage for the notes replay labeling script's pure logic."""

from __future__ import annotations

from collections import Counter

import pytest
from label_notes_replay import (
    Candidate,
    Judgment,
    LogRow,
    contains_hangul,
    join_labels,
    parse_label,
    sample_half,
)


def make_row(log_id: int, shape: str, *, hit_ids: tuple[str, ...] = ()) -> LogRow:
    return LogRow(
        log_id=log_id,
        query=f"query {log_id}",
        hit_ids=hit_ids,
        namespaces=None,
        shape=shape,
    )


def test_sample_half_round_robins_shapes_with_per_shape_cap():
    rows = [make_row(index, "keyword") for index in range(20)]
    rows += [make_row(100 + index, "free_text") for index in range(20)]
    rows += [make_row(200 + index, "cron_desire_tick") for index in range(20)]
    rows += [make_row(300 + index, "cron_other") for index in range(5)]

    picked = sample_half(rows, target=50)

    assert [row.shape for row in picked[:4]] == [
        "cron_desire_tick",
        "cron_other",
        "keyword",
        "free_text",
    ]
    assert Counter(row.shape for row in picked) == {
        "cron_desire_tick": 13,
        "cron_other": 5,
        "keyword": 13,
        "free_text": 19,
    }


def test_sample_half_stops_short_when_the_pool_runs_dry():
    rows = [make_row(index, "keyword") for index in range(3)]

    assert len(sample_half(rows, target=50)) == 3


def test_parse_label_validates_shape_and_candidate_range():
    assert parse_label({"relevant": [1, 3]}, 3) == [1, 3]
    assert parse_label({"relevant": [0, 2, 5, True]}, 3) == [2]
    assert parse_label({"relevant": []}, 3) == []
    assert parse_label({"relevant": "all"}, 3) is None
    assert parse_label({}, 3) is None
    assert parse_label(None, 3) is None


def _candidate_record(log_id: int, *ids: str) -> dict:
    return {
        "log_id": log_id,
        "query": f"query {log_id}",
        "shape": "keyword",
        "hit_ids": [],
        "namespaces": None,
        "candidates": [{"id": chunk_id, "score": 0.5, "text": "t"} for chunk_id in ids],
    }


def test_join_labels_maps_candidate_numbers_to_chunk_ids():
    judgments = join_labels(
        [_candidate_record(1, "a", "b", "c"), _candidate_record(2, "x")],
        [{"log_id": 1, "relevant": [1, 3]}, {"log_id": 2, "relevant": []}],
    )
    assert [j.relevant_ids for j in judgments] == [frozenset({"a", "c"}), frozenset()]
    assert [j.label_class for j in judgments] == ["A", "B"]


def test_join_labels_refuses_a_missing_or_malformed_label():
    with pytest.raises(ValueError, match="log_id: 1, 2"):
        join_labels(
            [_candidate_record(1, "a"), _candidate_record(2, "b"), _candidate_record(3, "c")],
            [{"log_id": 2, "relevant": "all"}, {"log_id": 3, "relevant": [1]}],
        )


def test_contains_hangul_detects_hangul_syllables():
    assert contains_hangul("postgres 데이터는 날아간거같은데")
    assert not contains_hangul("plain english query")


def test_judgment_derivations():
    row = make_row(1, "keyword", hit_ids=("a", "b", "c"))
    judgment = Judgment(
        row=row,
        candidates=(Candidate("a", 1.0, "text"),),
        relevant_ids=frozenset({"a", "z"}),
    )
    assert judgment.label_class == "with_hits"
    assert judgment.korean is False
    assert judgment.original_hit_overlap == 1 / 3

    empty_hit = Judgment(
        row=make_row(2, "keyword"),
        candidates=(),
        relevant_ids=frozenset({"a"}),
    )
    assert empty_hit.label_class == "A"
    assert empty_hit.original_hit_overlap == 0.0


def test_write_report_and_spotcheck_render_every_class(tmp_path):
    from label_notes_replay import write_report, write_spotcheck

    judgments = [
        Judgment(make_row(1, "keyword"), (Candidate("a", 1.0, "t"),), frozenset({"a"})),
        Judgment(make_row(2, "free_text"), (Candidate("b", 1.0, "t"),), frozenset()),
        Judgment(
            make_row(3, "keyword", hit_ids=("c",)), (Candidate("c", 1.0, "t"),), frozenset({"c"})
        ),
    ]
    write_report(judgments, tmp_path / "report.md")
    write_spotcheck(judgments, tmp_path / "spotcheck.md")

    report = (tmp_path / "report.md").read_text()
    assert "A          1        50.0" in report
    assert "mean original_hits_judged_relevant: 1.000" in report
    assert (tmp_path / "spotcheck.md").read_text().count("## log ") == 3
