"""Pure unit pins for memory_base.serve.notes.build_note_row.

These pins mirror the non-integration cases in tests/test_save_memory.py at
its new home. No DB/network involved.

Collection fails today: memory_base.serve.notes does not exist yet.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api, notes
from memory_base.serve.notes import build_note_row, save_note

NOW = 1_700_000_000.0
ID_RE = re.compile(r"^note:[0-9a-f]{16}$")

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


# ---- id scheme --------------------------------------------------------


def test_same_content_same_id():
    a = build_note_row("prefer ruff for linting", "note", ["test"], NOW)
    b = build_note_row("prefer ruff for linting", "note", ["test"], NOW)
    assert a["id"] == b["id"]


def test_different_content_different_id():
    a = build_note_row("prefer ruff for linting", "note", ["test"], NOW)
    b = build_note_row("prefer black for formatting", "note", ["test"], NOW)
    assert a["id"] != b["id"]


def test_id_format_note_prefix_16_hex():
    row = build_note_row("use pgvector halfvec for embeddings", "note", ["test"], NOW)
    assert ID_RE.match(row["id"]), row["id"]


# ---- id scheme: namespace qualification ------------------------------------


def test_default_namespace_id_is_byte_identical_to_legacy_format():
    """No namespace arg and explicit namespace='default' must produce the same id
    as before namespaces existed, so pre-existing rows keep their identity."""
    omitted = build_note_row("distilled content", "note", ["test"], NOW)
    explicit_default = build_note_row("distilled content", "note", ["test"], NOW, "default")
    assert omitted["id"] == explicit_default["id"]
    assert ID_RE.match(omitted["id"])


def test_non_default_namespace_id_is_namespace_qualified():
    row = build_note_row("distilled content", "note", ["test"], NOW, "team-a")
    assert row["id"].startswith("note:team-a:")


def test_same_content_different_namespace_different_id():
    default_row = build_note_row("distilled content", "note", ["test"], NOW, "default")
    team_row = build_note_row("distilled content", "note", ["test"], NOW, "team-a")
    assert default_row["id"] != team_row["id"]


def test_same_content_same_namespace_same_id():
    a = build_note_row("distilled content", "note", ["test"], NOW, "team-a")
    b = build_note_row("distilled content", "note", ["test"], NOW, "team-a")
    assert a["id"] == b["id"]


# ---- row shape ----------------------------------------------------------


def test_row_shape_exact_keys_no_embedding():
    row = build_note_row("distilled memory content", "note", ["test"], NOW)
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
    row = build_note_row(content, "decision", ["test"], NOW)
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


# ---- REST: author is required and allowlisted ------------------------------


async def _fake_save_note(
    content,
    *,
    tags,
    kind="note",
    supersedes=None,
    namespace="default",
    occurred_at=None,
    author=None,
):
    return {
        "id": "note:aaaaaaaaaaaaaaaa",
        "kind": kind,
        "stored": True,
        "superseded": None,
        "similar": [],
        "author": author,
    }


@pytest.mark.parametrize("author", [None, "", "   ", 123, ["natsume"]])
def test_save_memory_missing_author_400(monkeypatch, author):
    monkeypatch.setattr(api, "save_note", _fake_save_note)
    body = {"content": "distilled note text"}
    if author is not None:
        body["author"] = author
    response = client.post("/save_memory", json=body)
    assert response.status_code == 400
    assert response.json()["error"] == "author is required"


def test_save_memory_author_outside_the_allowlist_403(monkeypatch):
    monkeypatch.setattr(api, "save_note", _fake_save_note)
    response = client.post(
        "/save_memory", json={"author": "mallory", "content": "distilled note text"}
    )
    assert response.status_code == 403
    assert response.json()["error"] == "author 'mallory' is not permitted for this key"


def test_save_memory_forwards_the_author_to_save_note(monkeypatch):
    captured = {}

    async def fake_save_note(content, **kwargs):
        captured.update(kwargs)
        return {
            "id": "note:aaaaaaaaaaaaaaaa",
            "kind": "note",
            "stored": True,
            "superseded": None,
            "similar": [],
        }

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory", json={"author": "claude-code", "content": "distilled note text"}
    )
    assert response.status_code == 200
    assert captured["author"] == "claude-code"


# ---- save_note namespace gating (no DB/network) ----------------------------


class FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(self, registered: bool):
        self._registered = registered
        self.insert_args: tuple | None = None
        self.insert_calls: list[tuple] = []
        self._inserted_ids: set[str] = set()

    def transaction(self):
        return FakeTransaction()

    async def fetchval(self, query, *args):
        if "namespaces" in query:
            return self._registered
        return True

    async def execute(self, query, *args):
        if "INSERT INTO" in query:
            self.insert_args = args
            self.insert_calls.append(args)
            note_id = args[0]
            if note_id in self._inserted_ids:
                return "INSERT 0 0"
            self._inserted_ids.add(note_id)
            return "INSERT 0 1"
        return "UPDATE 1"

    async def fetch(self, query, *args):
        return []


def _patch_note_deps(monkeypatch, conn):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def fake_embed_text(embedder, text):
        return "[0]"

    monkeypatch.setattr(notes.db, "acquire", acquire)
    monkeypatch.setattr(notes, "embed_text", fake_embed_text)
    # VllmEmbedder() is constructed eagerly as an argument to embed_text, so it
    # must be faked too: its real constructor reaches EMB_URL, which no unit
    # test/CI environment configures.
    monkeypatch.setattr(notes, "VllmEmbedder", lambda: None)
    monkeypatch.setattr(notes, "ensure_schema_once", _noop)


async def _noop(conn):
    return None


def test_save_note_rejects_unregistered_namespace(monkeypatch):
    conn = FakeConnection(registered=False)
    _patch_note_deps(monkeypatch, conn)
    with pytest.raises(ValueError, match="unregistered namespace"):
        asyncio.run(save_note("distilled content", tags=["test"], namespace="ghost"))
    assert conn.insert_args is None


def test_save_note_stamps_namespace_column_on_insert(monkeypatch):
    conn = FakeConnection(registered=True)
    _patch_note_deps(monkeypatch, conn)
    asyncio.run(save_note("distilled content", tags=["test"], namespace="team-a"))
    assert conn.insert_args is not None
    assert "team-a" in conn.insert_args


def test_save_note_defaults_to_default_namespace(monkeypatch):
    conn = FakeConnection(registered=True)
    _patch_note_deps(monkeypatch, conn)
    asyncio.run(save_note("distilled content", tags=["test"]))
    assert "default" in conn.insert_args


# ---- cross-namespace independence: no silent no-op on the second namespace -


def test_save_note_same_content_two_namespaces_both_stored(monkeypatch):
    conn = FakeConnection(registered=True)
    _patch_note_deps(monkeypatch, conn)
    content = "distilled content shared across namespaces"
    result_default = asyncio.run(save_note(content, tags=["test"], namespace="default"))
    result_team = asyncio.run(save_note(content, tags=["test"], namespace="team-a"))
    assert result_default["stored"] is True
    assert result_team["stored"] is True
    assert result_default["id"] != result_team["id"]
    assert len(conn.insert_calls) == 2


def test_save_note_same_content_same_namespace_still_dedups(monkeypatch):
    conn = FakeConnection(registered=True)
    _patch_note_deps(monkeypatch, conn)
    content = "distilled content repeated in one namespace"
    first = asyncio.run(save_note(content, tags=["test"], namespace="team-a"))
    second = asyncio.run(save_note(content, tags=["test"], namespace="team-a"))
    assert first["stored"] is True
    assert second["stored"] is False
    assert first["id"] == second["id"]
