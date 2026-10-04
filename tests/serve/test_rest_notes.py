"""Unit pins for the note id scheme, the note row, the author gate, and namespace gating.

No DB/network involved.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.notes import store
from memory_base.serve.notes.store import build_note_row, save_note

NOW = 1_700_000_000.0
ID_RE = re.compile(r"^note:default:[0-9a-f]{16}$")

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


# ---- id scheme --------------------------------------------------------


def test_different_content_different_id():
    a = build_note_row("prefer ruff for linting", "work", ["test"], NOW)
    b = build_note_row("prefer black for formatting", "work", ["test"], NOW)
    assert a["id"] != b["id"]


def test_id_is_the_namespace_and_sha256_of_the_stripped_content():
    digest = hashlib.sha256(b"prefer ruff for linting").hexdigest()[:16]
    row = build_note_row("  prefer ruff for linting\n", "work", ["test"], NOW, "team-a")
    assert row["id"] == f"note:team-a:{digest}"


def test_default_namespace_id_is_namespace_qualified_like_every_other():
    omitted = build_note_row("distilled content", "work", ["test"], NOW)
    explicit_default = build_note_row("distilled content", "work", ["test"], NOW, "default")
    assert omitted["id"] == explicit_default["id"]
    assert ID_RE.match(omitted["id"])


def test_same_content_different_namespace_different_id():
    default_row = build_note_row("distilled content", "work", ["test"], NOW, "default")
    team_row = build_note_row("distilled content", "work", ["test"], NOW, "team-a")
    assert default_row["id"] != team_row["id"]


# ---- row shape ----------------------------------------------------------


def test_row_shape_exact_keys_no_embedding():
    row = build_note_row("distilled memory content", "work", ["test"], NOW)
    assert set(row) == {
        "id",
        "source_type",
        "source_ref",
        "kind",
        "session_id",
        "raw",
        "distilled",
        "timestamp",
        "metadata",
        "occurred_at",
    }
    assert "embedding" not in row


def test_row_field_values():
    content = "the burst gate uses a weighted signal sum"
    row = build_note_row(content, "work", ["test"], NOW)
    assert row["source_type"] == "agent_note"
    assert row["source_ref"] == "save_memory"
    assert row["kind"] == "work"
    assert row["session_id"] == row["id"]
    assert row["raw"] == content
    assert row["distilled"] == content
    assert row["timestamp"] == NOW


def test_tags_land_in_metadata():
    row = build_note_row("content with tags", "work", ["infra", "db"], NOW)
    assert row["metadata"] == {"tags": ["infra", "db"]}


# ---- REST: author is required and allowlisted ------------------------------


async def _fake_save_note(
    content,
    *,
    tags,
    kind,
    supersedes=None,
    namespace="default",
    occurred_at=None,
    author=None,
    allow_similar=False,
):
    return {
        "id": "note:aaaaaaaaaaaaaaaa",
        "kind": kind,
        "stored": True,
        "superseded": None,
        "similar": [],
        "author": author,
    }


@pytest.mark.parametrize("author", [None, "   "])
def test_save_memory_missing_author_400(monkeypatch, author):
    monkeypatch.setattr(store, "save_note", _fake_save_note)
    body = {"content": "distilled note text", "kind": "work"}
    if author is not None:
        body["author"] = author
    response = client.post("/save_memory", json=body)
    assert response.status_code == 400
    assert response.json()["error"] == "author is required"


def test_save_memory_author_outside_the_allowlist_403(monkeypatch):
    monkeypatch.setattr(store, "save_note", _fake_save_note)
    response = client.post(
        "/save_memory", json={"author": "mallory", "content": "distilled note text", "kind": "work"}
    )
    assert response.status_code == 403
    assert response.json()["error"] == "author 'mallory' is not permitted for this key"


def test_save_memory_forwards_the_author_to_save_note(monkeypatch):
    captured = {}

    async def fake_save_note(content, **kwargs):
        captured.update(kwargs)
        return {
            "id": "note:aaaaaaaaaaaaaaaa",
            "kind": "work",
            "stored": True,
            "superseded": None,
            "similar": [],
        }

    monkeypatch.setattr(store, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={"author": "claude-code", "content": "distilled note text", "kind": "work"},
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

    def transaction(self):
        return FakeTransaction()

    async def fetchval(self, query, *args):
        if "namespaces" in query:
            return self._registered
        if "chunk_kind" in query:
            return "work"
        return True

    async def execute(self, query, *args):
        if "INSERT INTO" in query:
            self.insert_args = args
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

    monkeypatch.setattr(store.db, "acquire", acquire)
    monkeypatch.setattr(store, "embed_text", fake_embed_text)
    # VllmEmbedder() is constructed eagerly as an argument to embed_text, so it
    # must be faked too: its real constructor reaches EMB_URL, which no unit
    # test/CI environment configures.
    monkeypatch.setattr(store, "VllmEmbedder", lambda: None)
    monkeypatch.setattr(store, "ensure_schema_once", _noop)


async def _noop(conn):
    return None


def test_save_note_rejects_unregistered_namespace(monkeypatch):
    conn = FakeConnection(registered=False)
    _patch_note_deps(monkeypatch, conn)
    with pytest.raises(ValueError, match="unregistered namespace"):
        asyncio.run(save_note("distilled content", tags=["test"], kind="work", namespace="ghost"))
    assert conn.insert_args is None


def test_save_note_stamps_namespace_column_on_insert(monkeypatch):
    conn = FakeConnection(registered=True)
    _patch_note_deps(monkeypatch, conn)
    asyncio.run(save_note("distilled content", tags=["test"], kind="work", namespace="team-a"))
    assert conn.insert_args is not None
    assert "team-a" in conn.insert_args


def test_save_note_defaults_to_default_namespace(monkeypatch):
    conn = FakeConnection(registered=True)
    _patch_note_deps(monkeypatch, conn)
    asyncio.run(save_note("distilled content", tags=["test"], kind="work"))
    assert "default" in conn.insert_args
