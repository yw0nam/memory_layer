"""Contract tests for the save_memory MCP tool (red-first).

Pure tests pin build_note_row's id scheme, row shape, and validation with no
DB/embedder/network. Integration tests (marked ``integration``, skipped when the
DB is unreachable) call the real tool through the in-process REST app
(``rest_in_process`` fixture) against Postgres + embedder and clean up every
row they insert so reruns stay stable.
"""

from __future__ import annotations

import asyncio
import re

import pytest

import asyncpg

from memory_base.common import DB_URL, PG_SCHEMA
from memory_base.retrieval.search import search
from memory_base.serve.mcp_server import save_memory
from memory_base.serve.notes import build_note_row

NOW = 1_700_000_000.0
ID_RE = re.compile(r"^note:[0-9a-f]{16}$")


# ---- pure: id scheme -------------------------------------------------------


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


# ---- pure: row shape -------------------------------------------------------


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


def test_tags_are_normalized_and_deduplicated():
    row = build_note_row(
        "content with normalized tags",
        "note",
        [" Infrastructure ", "DATABASE", "infrastructure", "  "],
        NOW,
    )
    assert row["metadata"] == {"tags": ["infrastructure", "database"]}


# ---- pure: validation ------------------------------------------------------


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


@pytest.mark.parametrize("tags", ["infra", {"infra": True}, [1], ["infra", None]])
def test_malformed_tags_rejected(tags):
    with pytest.raises(ValueError, match="tags must be a list of strings"):
        build_note_row("valid content", "note", tags, NOW)


# ---- integration: real DB + embedder --------------------------------------


def _db_reachable() -> bool:
    async def _check() -> None:
        conn = await asyncpg.connect(DB_URL, timeout=5)
        await conn.close()

    try:
        asyncio.run(_check())
        return True
    except Exception:
        return False


_DB = _db_reachable()
requires_db = pytest.mark.skipif(not _DB, reason=f"DB not reachable at {DB_URL}")


async def _delete(note_id: str) -> None:
    conn = await asyncpg.connect(DB_URL)
    try:
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', note_id)
    finally:
        await conn.close()


async def _fetchrow(note_id: str):
    conn = await asyncpg.connect(DB_URL)
    try:
        return await conn.fetchrow(
            f'SELECT * FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', note_id
        )
    finally:
        await conn.close()


async def _count(note_id: str) -> int:
    conn = await asyncpg.connect(DB_URL)
    try:
        return await conn.fetchval(
            f'SELECT count(*) FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', note_id
        )
    finally:
        await conn.close()


@pytest.mark.integration
@requires_db
def test_save_memory_stores_row_in_db(rest_in_process):
    content = "save_memory integration: notes are stored without LLM distillation"
    note_id = build_note_row(content, "note", None, NOW)["id"]
    asyncio.run(_delete(note_id))
    try:
        result = asyncio.run(save_memory(content, kind="note", tags=["pytest"]))
        assert result["id"] == note_id
        assert result["stored"] is True

        row = asyncio.run(_fetchrow(note_id))
        assert row is not None
        assert row["source_type"] == "agent_note"
        assert row["source_ref"] == "save_memory"
        assert row["chunk_kind"] == "note"
        assert row["session_id"] == note_id
        assert row["content_raw"] == content
        assert row["distilled"] == content
        assert row["idf_score"] is None
    finally:
        asyncio.run(_delete(note_id))


@pytest.mark.integration
@requires_db
def test_save_memory_duplicate_is_noop(rest_in_process):
    content = "save_memory integration: re-saving identical content is idempotent"
    note_id = build_note_row(content, "note", None, NOW)["id"]
    asyncio.run(_delete(note_id))
    try:
        first = asyncio.run(save_memory(content))
        second = asyncio.run(save_memory(content))
        assert first["stored"] is True
        assert second["stored"] is False
        assert asyncio.run(_count(note_id)) == 1
    finally:
        asyncio.run(_delete(note_id))


@pytest.mark.integration
@requires_db
def test_saved_note_found_by_search(rest_in_process):
    content = "save_memory integration: pgvector halfvec powers hybrid retrieval search"
    note_id = build_note_row(content, "note", None, NOW)["id"]
    asyncio.run(_delete(note_id))
    try:
        asyncio.run(save_memory(content))
        hits = asyncio.run(search(content, source="history", rerank=False))
        assert any(h.meta.get("id") == note_id for h in hits)
    finally:
        asyncio.run(_delete(note_id))
