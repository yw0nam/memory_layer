"""Pure unit pins for memory_base.serve.admin.is_cold.

Per docs/specs/impl_lifecycle.md §4, the cold-tier candidate rule is:

    ts_last_active < now - age_days*86400 AND
    COALESCE(last_hit_at, ts_last_active) < now - unhit_days*86400

pinned here as a pure predicate with signature
``is_cold(ts_last_active, last_hit_at, now, age_days, unhit_days) -> bool``.
No DB/network involved.

Collection fails today: memory_base.serve.admin does not exist yet.
"""

from __future__ import annotations

from memory_base.serve import admin

DAY = 86400.0
NOW = 1_700_000_000.0
AGE_DAYS = 180
UNHIT_DAYS = 90
AGE_CUTOFF = NOW - AGE_DAYS * DAY
UNHIT_CUTOFF = NOW - UNHIT_DAYS * DAY


def test_age_only_not_enough():
    # old enough by age, but hit recently -> not cold
    assert admin.is_cold(AGE_CUTOFF - DAY, NOW - DAY, NOW, AGE_DAYS, UNHIT_DAYS) is False


def test_unhit_only_not_enough():
    # unhit long enough, but not old enough by age -> not cold
    assert admin.is_cold(NOW - DAY, UNHIT_CUTOFF - DAY, NOW, AGE_DAYS, UNHIT_DAYS) is False


def test_both_conditions_true():
    assert admin.is_cold(AGE_CUTOFF - DAY, UNHIT_CUTOFF - DAY, NOW, AGE_DAYS, UNHIT_DAYS) is True


def test_last_hit_at_none_falls_back_to_ts_last_active_when_old_enough():
    # older than both windows, never hit -> cold via COALESCE fallback
    assert admin.is_cold(AGE_CUTOFF - DAY, None, NOW, AGE_DAYS, UNHIT_DAYS) is True


def test_recently_hit_old_row_not_cold():
    # old by age, but hit recently -> COALESCE picks last_hit_at, not cold
    assert admin.is_cold(AGE_CUTOFF - DAY, NOW - DAY, NOW, AGE_DAYS, UNHIT_DAYS) is False


def test_age_boundary_exactly_at_cutoff_is_not_cold():
    # "<" is strict: equal to the cutoff does not count as older
    assert admin.is_cold(AGE_CUTOFF, None, NOW, AGE_DAYS, UNHIT_DAYS) is False


def test_age_boundary_just_past_cutoff_is_cold():
    assert admin.is_cold(AGE_CUTOFF - 1, None, NOW, AGE_DAYS, UNHIT_DAYS) is True


def test_unhit_boundary_exactly_at_cutoff_is_not_cold():
    assert admin.is_cold(AGE_CUTOFF - DAY, UNHIT_CUTOFF, NOW, AGE_DAYS, UNHIT_DAYS) is False


def test_unhit_boundary_just_past_cutoff_is_cold():
    assert admin.is_cold(AGE_CUTOFF - DAY, UNHIT_CUTOFF - 1, NOW, AGE_DAYS, UNHIT_DAYS) is True
