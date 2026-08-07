"""Unit tests for document conversion, chunking, CSV cards, and row mapping."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from memory_base.adapters import document
from memory_base.adapters.document import Chunk


@pytest.mark.parametrize(
    "filename",
    [
        "a.md",
        "a.markdown",
        "a.txt",
        "a.rst",
        "a.html",
        "a.htm",
        "a.pdf",
        "a.docx",
        "a.pptx",
        "a.csv",
    ],
)
def test_extension_matrix_accepts_exact_supported_formats(filename):
    assert document.extension_for(filename) == Path(filename).suffix


@pytest.mark.parametrize("filename", ["a.doc", "a.xls", "a.ppt", "a", "a.json", "a.md.exe"])
def test_extension_matrix_rejects_legacy_and_unknown_formats(filename):
    with pytest.raises(document.UnsupportedDocumentError):
        document.extension_for(filename)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Guide.MD", "guide.md"),
        ("folder/Guide.MD", "guide.md"),
        (r"folder\Guide.MD", "guide.md"),
        ("a_b-c.1.txt", "a_b-c.1.txt"),
    ],
)
def test_document_id_normalization(value, expected):
    assert document.normalize_document_id(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", ":bad", "bad:id", "-bad", "a/bad:id", "a" * 122, "white space.md"],
)
def test_document_id_validation_rejects_collision_and_pattern_cases(value):
    with pytest.raises(document.DocumentError):
        document.normalize_document_id(value)


def test_chunker_tracks_heading_path_and_keeps_fence_atomic():
    prose = "Alpha prose " * 30
    code = "```\n" + "\n".join(f"line {index}" for index in range(40)) + "\n```"
    result = document.chunk_markdown(f"# Root\n\n{prose}\n\n## Child\n\n{code}")
    assert [chunk.heading_path for chunk in result.chunks] == [("Root",), ("Root", "Child")]
    assert result.chunks[1].text == code
    assert all(200 <= len(chunk.text) <= 2_000 for chunk in result.chunks)


def test_chunker_splits_five_thousand_character_paragraph_at_whitespace():
    result = document.chunk_markdown("word " * 1_000)
    assert len(result.chunks) == 3
    assert all(200 <= len(chunk.text) <= 2_000 for chunk in result.chunks)
    assert "".join(chunk.text.replace(" ", "") for chunk in result.chunks) == "word" * 1_000


def test_residue_merges_into_predecessor_near_hard_bound():
    chunks, dropped = document._residue_pass([Chunk("a" * 1_900, ()), Chunk("b" * 98, ())])
    assert dropped == 0
    assert len(chunks) == 1
    assert len(chunks[0].text) == 2_000


def test_residue_merges_into_successor_near_hard_bound():
    chunks, dropped = document._residue_pass([Chunk("a" * 98, ()), Chunk("b" * 1_900, ())])
    assert dropped == 0
    assert len(chunks) == 1
    assert len(chunks[0].text) == 2_000


def test_unmergeable_residue_is_dropped():
    chunks, dropped = document._residue_pass(
        [Chunk("a" * 1_950, ("A",)), Chunk("b" * 100, ("A",)), Chunk("c" * 300, ("B",))]
    )
    assert [chunk.text[0] for chunk in chunks] == ["a", "c"]
    assert dropped == 1


def test_mixed_heading_uses_matching_successor_when_both_sizes_fit():
    chunks, dropped = document._residue_pass(
        [Chunk("a" * 500, ("A",)), Chunk("b" * 100, ("B",)), Chunk("c" * 500, ("B",))]
    )
    assert dropped == 0
    assert [chunk.heading_path for chunk in chunks] == [("A",), ("B",)]
    assert chunks[1].text.startswith("b")


def test_predecessor_merge_reconsiders_consecutive_residues():
    chunks, dropped = document._residue_pass(
        [Chunk("a" * 1_700, ()), Chunk("b" * 100, ()), Chunk("c" * 100, ())]
    )
    assert dropped == 0
    assert len(chunks) == 1
    assert len(chunks[0].text) == 1_904


def test_successor_merge_under_two_hundred_is_reconsidered_and_dropped():
    chunks, dropped = document._residue_pass(
        [Chunk("a" * 50, ("A",)), Chunk("b" * 50, ("A",)), Chunk("c" * 500, ("B",))]
    )
    assert [chunk.text[0] for chunk in chunks] == ["c"]
    assert dropped == 1


def test_residue_at_heading_path_end_does_not_cross_heading():
    chunks, dropped = document._residue_pass(
        [Chunk("a" * 300, ("A",)), Chunk("b" * 100, ("B",)), Chunk("c" * 300, ("C",))]
    )
    assert [chunk.heading_path for chunk in chunks] == [("A",), ("C",)]
    assert dropped == 1


def test_junk_letter_ratio_uses_non_whitespace_denominator():
    assert document.is_junk(("a" * 59) + ("1" * 141) + (" " * 500))
    assert not document.is_junk(("a" * 60) + ("1" * 140) + (" " * 500))


def test_junk_link_lines_require_span_covering_more_than_half_the_line():
    link = "[documentation](https://example.com/reference)"
    assert document.is_junk("\n".join([link, link, "Useful prose about the referenced API."]))
    assert not document.is_junk(
        "\n".join(
            [f"Read {link} for supporting detail and examples.", "Useful prose.", "More prose."]
        )
    )


def test_ordinals_are_contiguous_after_junk_drops():
    result = document.chunk_markdown(("a" * 800) + "\n\n" + ("1" * 800) + "\n\n" + ("b" * 800))
    assert [chunk.ordinal for chunk in result.chunks] == [0, 1]
    assert [chunk.text[0] for chunk in result.chunks] == ["a", "b"]
    assert result.dropped == 1


def test_document_row_ids_metadata_and_search_refs():
    chunks = [Chunk("Body text " * 30, ("Root", "Child"), 0)]
    rows = document.map_document_rows(
        chunks,
        tags=["project atlas"],
        filename="guide.md",
        document_id="guide.md",
        content_hash="abc",
        format_name="md",
        converter="markitdown:0.1.6",
        origin="drive:item",
        timestamp=1.0,
    )
    (row,) = rows
    assert row["id"] == "doc:guide.md:0"
    assert row["chunk_kind"] == "doc"
    assert row["source_ref"] == "guide.md"
    assert row["content_raw"] == chunks[0].text
    assert row["embedding_text"] == f"Root > Child\n\n{chunks[0].text}"
    assert row["metadata"]["search_ref"] == "guide.md#chunk-0"
    assert row["metadata"]["heading_path"] == ["Root", "Child"]
    assert row["metadata"]["ordinal"] == 0
    assert row["metadata"]["content_hash"] == "abc"
    assert row["metadata"]["tags"] == ["project atlas"]
    assert row["namespace"] == "default"


def test_document_rows_stamp_explicit_namespace():
    chunks = [Chunk("Body text " * 30, ("Root",), 0)]
    (row,) = document.map_document_rows(
        chunks,
        tags=[],
        filename="guide.md",
        document_id="guide.md",
        content_hash="abc",
        format_name="md",
        converter="markitdown:0.1.6",
        origin=None,
        timestamp=1.0,
        namespace="team-a",
    )
    assert row["namespace"] == "team-a"
    # non-default namespace qualifies the id so it never collides with another
    # namespace's row for the same document_id (default keeps the legacy id).
    assert row["id"] == "doc:team-a:guide.md:0"


def test_document_rows_same_document_id_different_namespaces_have_disjoint_ids():
    chunks = [Chunk("Body text " * 30, ("Root",), 0)]
    kwargs = dict(
        tags=[],
        filename="guide.md",
        document_id="guide.md",
        content_hash="abc",
        format_name="md",
        converter="markitdown:0.1.6",
        origin=None,
        timestamp=1.0,
    )
    default_rows = document.map_document_rows(chunks, **kwargs, namespace="default")
    team_a_rows = document.map_document_rows(chunks, **kwargs, namespace="team-a")
    team_b_rows = document.map_document_rows(chunks, **kwargs, namespace="team-b")
    all_ids = [row["id"] for rows in (default_rows, team_a_rows, team_b_rows) for row in rows]
    assert len(all_ids) == len(set(all_ids))
    # default keeps the pre-namespace id format exactly.
    assert default_rows[0]["id"] == "doc:guide.md:0"


def test_csv_sample_and_card_builder_use_header_first_twenty_rows(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,value\n" + "\n".join(f"row-{i},{i}" for i in range(25)))
    sample = document.read_csv_sample(path)
    assert sample.header == ["name", "value"]
    assert len(sample.rows) == 20
    assert sample.row_count == 25
    assert sample.column_count == 2

    captured = {}

    async def summarize(text, context):
        captured["text"] = text
        captured["context"] = context
        return {"summary": "A sampled value table.", "tags": ["sample table"]}

    card = asyncio.run(document.build_csv_card(sample, summarize))
    assert card["summary"] == "A sampled value table."
    assert "Total data rows: 25" in captured["text"]
    assert "row-19,19" in captured["text"]
    assert "row-20,20" not in captured["text"]


def test_csv_card_row_mapping():
    sample = document.CSVSample(["name"], [["one"]], 1, 1)
    row = document.map_csv_card_row(
        {"summary": "A one-column name table.", "tags": ["name table"]},
        sample,
        filename="names.csv",
        document_id="names.csv",
        content_hash="hash",
        origin=None,
        timestamp=2.0,
    )
    assert row["id"] == "doc:names.csv:card:0"
    assert row["content_raw"] == row["distilled"]
    assert row["metadata"]["search_ref"] == "names.csv#card-0"
    assert row["metadata"]["row_count"] == 1
    assert row["namespace"] == "default"


def test_csv_card_row_same_document_id_different_namespaces_have_disjoint_ids():
    sample = document.CSVSample(["name"], [["one"]], 1, 1)
    card = {"summary": "A one-column name table.", "tags": ["name table"]}
    kwargs = dict(
        filename="names.csv",
        document_id="names.csv",
        content_hash="hash",
        origin=None,
        timestamp=2.0,
    )
    default_row = document.map_csv_card_row(card, sample, **kwargs, namespace="default")
    team_a_row = document.map_csv_card_row(card, sample, **kwargs, namespace="team-a")
    assert default_row["id"] == "doc:names.csv:card:0"
    assert team_a_row["id"] == "doc:team-a:names.csv:card:0"
    assert default_row["id"] != team_a_row["id"]


def test_csv_size_cap_is_enforced_before_reading(tmp_path):
    path = tmp_path / "large.csv"
    path.write_bytes(b"x" * (document.CSV_MAX_BYTES + 1))
    with pytest.raises(document.DocumentError, match="5 MB"):
        document.read_csv_sample(path)
