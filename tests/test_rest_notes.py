"""Pure unit pins for memory_base.serve.notes.build_note_row.

These pins mirror the non-integration cases in tests/test_save_memory.py at
its new home. No DB/network involved.

Collection fails today: memory_base.serve.notes does not exist yet.
"""

from __future__ import annotations

import re

import pytest

from memory_base.serve.notes import build_note_row

NOW = 1_700_000_000.0
ID_RE = re.compile(r"^note:[0-9a-f]{16}$")


# ---- id scheme --------------------------------------------------------


def test_same_content_same_id():
    a = build_note_row("prefer ruff for linting", "note", None, NOW)
    b = build_note_row("prefer ruff for linting", "note", None, NOW)
    assert a["id"] == b["id"]


def test_different_content_different_id():
    a = build_note_row("prefer ruff for linting", "note", None, NOW)
    b = build_note_row("prefer black for formatting", "note", None, NOW)
    assert a["id"] != b["id"]


def test_id_format_note_prefix_16_hex():
    row = build_note_row("use pgvector halfvec for embeddings", "note", None, NOW)
    assert ID_RE.match(row["id"]), row["id"]


# ---- row shape ----------------------------------------------------------


def test_row_shape_exact_keys_no_embedding():
    row = build_note_row("distilled memory content", "note", None, NOW)
    assert set(row) == {
        "id",
        "source_type",
        "source_ref",
        "kind",
        "session_id",
        "raw",
        "distilled",
        "timestamp",
        "idf",
        "metadata",
    }
    assert "embedding" not in row


def test_row_field_values():
    content = "the burst gate uses a weighted signal sum"
    row = build_note_row(content, "decision", None, NOW)
    assert row["source_type"] == "agent_note"
    assert row["source_ref"] == "save_memory"
    assert row["kind"] == "decision"
    assert row["session_id"] == row["id"]
    assert row["raw"] == content
    assert row["distilled"] == content
    assert row["timestamp"] == NOW
    assert row["idf"] is None


def test_tags_land_in_metadata():
    row = build_note_row("content with tags", "note", ["infra", "db"], NOW)
    assert row["metadata"] == {"tags": ["infra", "db"]}


def test_no_tags_empty_metadata():
    row = build_note_row("content without tags", "note", None, NOW)
    assert row["metadata"] == {}


# ---- validation ---------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_empty_or_whitespace_content_rejected(bad):
    with pytest.raises(ValueError):
        build_note_row(bad, "note", None, NOW)


def test_oversized_content_rejected():
    with pytest.raises(ValueError):
        build_note_row("x" * 4001, "note", None, NOW)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        build_note_row("valid content", "reminder", None, NOW)
