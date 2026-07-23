"""Unit tests for legacy metadata tag normalization."""

from memory_base.schema import normalize_legacy_metadata


def test_legacy_string_tags_are_removed():
    assert normalize_legacy_metadata({"tags": "infra", "keep": 1}) == {"keep": 1}


def test_legacy_object_tags_are_removed():
    assert normalize_legacy_metadata({"tags": {"name": "infra"}, "keep": 1}) == {"keep": 1}


def test_legacy_mixed_array_keeps_normalized_unique_strings():
    metadata = {"tags": [" Infra ", 3, "DATABASE", "", None, "infra"], "keep": 1}
    assert normalize_legacy_metadata(metadata) == {
        "tags": ["infra", "database"],
        "keep": 1,
    }


def test_legacy_empty_array_removes_tags_key():
    assert normalize_legacy_metadata({"tags": [], "keep": 1}) == {"keep": 1}


def test_legacy_missing_tags_key_is_unchanged():
    metadata = {"keep": 1}
    assert normalize_legacy_metadata(metadata) is metadata
