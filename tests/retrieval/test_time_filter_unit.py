"""Unit tests for the since/until time filter: parsing, validation, SQL predicates."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory_base.retrieval.search import (
    history_predicates,
    normalize_time_range,
    parse_time_bound,
    validate_search_options,
)

DAY = 86400.0
AUG_12 = datetime(2026, 8, 12, tzinfo=timezone.utc).timestamp()


# ---- parse_time_bound -------------------------------------------------------


def test_bare_date_parses_to_utc_midnight():
    assert parse_time_bound("2026-08-12") == AUG_12


def test_bare_date_end_bound_covers_the_whole_day():
    assert parse_time_bound("2026-08-12", end=True) == AUG_12 + DAY


def test_datetime_end_bound_is_the_exact_instant():
    assert parse_time_bound("2026-08-12T06:30:00", end=True) == AUG_12 + 6.5 * 3600


def test_naive_datetime_is_read_as_utc():
    assert parse_time_bound("2026-08-12T00:00:00") == AUG_12


def test_timezone_offset_is_respected():
    assert parse_time_bound("2026-08-12T09:00:00+09:00") == AUG_12


@pytest.mark.parametrize("bad", ["not-a-date", "", "   ", None, 123, 1700000000.0])
def test_invalid_time_bound_rejected(bad):
    with pytest.raises(ValueError):
        parse_time_bound(bad)


# ---- normalize_time_range ---------------------------------------------------


def test_omitted_bounds_normalize_to_none():
    assert normalize_time_range(None, None) == (None, None)


def test_single_bounds_pass_through():
    assert normalize_time_range("2026-08-12", None) == (AUG_12, None)
    assert normalize_time_range(None, "2026-08-12") == (None, AUG_12 + DAY)


def test_same_bare_date_covers_one_full_day():
    since, until = normalize_time_range("2026-08-12", "2026-08-12")
    assert until - since == DAY


def test_since_at_or_after_until_rejected():
    with pytest.raises(ValueError, match="earlier"):
        normalize_time_range("2026-08-13", "2026-08-12")
    with pytest.raises(ValueError, match="earlier"):
        normalize_time_range("2026-08-12T06:00:00", "2026-08-12T06:00:00")


# ---- history_predicates -----------------------------------------------------


def test_predicates_bound_ts_last_active():
    predicates, args = history_predicates(
        include_archived=True, kind=None, tags=None, since=100.0, until=200.0
    )
    assert predicates == "ts_last_active >= $2 AND ts_last_active < $3"
    assert args == [100.0, 200.0]


def test_time_clauses_number_after_the_other_filters():
    predicates, args = history_predicates(
        include_archived=False,
        kind="note",
        tags=["infra"],
        namespaces=["team-a"],
        since=1.0,
        until=2.0,
    )
    assert args == ["note", ["infra"], ["team-a"], 1.0, 2.0]
    assert "ts_last_active >= $5" in predicates
    assert "ts_last_active < $6" in predicates


def test_omitted_bounds_add_no_clause():
    predicates, args = history_predicates(include_archived=True, kind=None, tags=None)
    assert "ts_last_active" not in predicates
    assert args == []


# ---- validate_search_options ------------------------------------------------


def test_memory_source_accepts_time_bounds():
    kind, tags, repo, since, until = validate_search_options(
        "memory", None, None, since="2026-08-12", until="2026-08-13"
    )
    assert since == AUG_12
    assert until == AUG_12 + 2 * DAY


@pytest.mark.parametrize("source", ["code", "all"])
def test_time_bounds_require_memory_source(source):
    with pytest.raises(ValueError, match='source="memory"'):
        validate_search_options(source, None, None, since="2026-08-12")
    with pytest.raises(ValueError, match='source="memory"'):
        validate_search_options(source, None, None, until="2026-08-12")
