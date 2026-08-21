"""Contract tests for episode notes: the "episode" kind and occurred_at (red-first).

Pure/unit sections mirror tests/serve/test_rest_notes.py's FakeConnection pattern:
no DB/network involved. REST sections mirror tests/serve/test_rest_api.py (route
delegates to a monkeypatched ``api.save_note``). MCP sections mirror
tests/serve/test_supersede.py (httpx.MockTransport proxy).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import pytest
from starlette.testclient import TestClient

from memory_base.serve import api, notes
from memory_base.serve.mcp_server import save_memory
from memory_base.serve.notes import build_note_row, save_note

NOW = 1_700_000_000.0  # 2023-11-14T22:13:20Z

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


# ---- pure: episode is a valid kind -----------------------------------------


def test_episode_kind_accepted():
    row = build_note_row("met alice for lunch and discussed the roadmap", "episode", None, NOW)
    assert row["kind"] == "episode"


def test_unknown_kind_still_rejected():
    with pytest.raises(ValueError, match="kind must be one of"):
        build_note_row("valid content", "reminder", None, NOW)


# ---- save_note occurred_at gating (no DB/network) --------------------------


class FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(self, registered: bool = True):
        self._registered = registered
        self.insert_args: tuple | None = None

    def transaction(self):
        return FakeTransaction()

    async def fetchval(self, query, *args):
        if "namespaces" in query:
            return self._registered
        return True

    async def execute(self, query, *args):
        if "INSERT INTO" in query:
            self.insert_args = args
            return "INSERT 0 1"
        return "UPDATE 1"

    async def fetch(self, query, *args):
        return []


async def _noop(conn):
    return None


def _patch_note_deps(monkeypatch, conn):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def fake_embed_text(embedder, text):
        return "[0]"

    monkeypatch.setattr(notes.db, "acquire", acquire)
    monkeypatch.setattr(notes, "embed_text", fake_embed_text)
    monkeypatch.setattr(notes, "VllmEmbedder", lambda: None)
    monkeypatch.setattr(notes, "ensure_schema_once", _noop)


def test_occurred_at_sets_stored_timestamp(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    monkeypatch.setattr(notes.time, "time", lambda: NOW)
    asyncio.run(save_note("distilled content", occurred_at="2020-01-01"))
    expected = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    assert conn.insert_args[8] == expected


def test_occurred_at_omitted_uses_current_time(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    monkeypatch.setattr(notes.time, "time", lambda: NOW)
    asyncio.run(save_note("distilled content"))
    assert conn.insert_args[8] == NOW


def test_occurred_at_malformed_rejected(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    monkeypatch.setattr(notes.time, "time", lambda: NOW)
    with pytest.raises(ValueError):
        asyncio.run(save_note("distilled content", occurred_at="not-a-date"))
    assert conn.insert_args is None


def test_occurred_at_in_future_rejected(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    monkeypatch.setattr(notes.time, "time", lambda: NOW)
    with pytest.raises(ValueError, match="future"):
        asyncio.run(save_note("distilled content", occurred_at="2024-01-01"))
    assert conn.insert_args is None


def test_occurred_at_does_not_change_content_hash_id(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    monkeypatch.setattr(notes.time, "time", lambda: NOW)
    content = "backfilled episode: shipped the search fix"
    plain_id = build_note_row(content, "episode", None, NOW)["id"]
    result = asyncio.run(save_note(content, kind="episode", occurred_at="2020-01-01"))
    assert result["id"] == plain_id


# ---- REST: POST /save_memory forwards/validates occurred_at ----------------


def test_save_memory_forwards_occurred_at_to_save_note(monkeypatch):
    captured = {}

    async def fake_save_note(
        content, kind="note", tags=None, supersedes=None, namespace="default", occurred_at=None
    ):
        captured["occurred_at"] = occurred_at
        return {"id": "note:x", "kind": kind, "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory", json={"content": "distilled text", "occurred_at": "2023-06-01"}
    )
    assert response.status_code == 200
    assert captured["occurred_at"] == "2023-06-01"


def test_save_memory_omitted_occurred_at_forwards_none(monkeypatch):
    captured = {}

    async def fake_save_note(
        content, kind="note", tags=None, supersedes=None, namespace="default", occurred_at=None
    ):
        captured["occurred_at"] = occurred_at
        return {"id": "note:x", "kind": kind, "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post("/save_memory", json={"content": "distilled text"})
    assert response.status_code == 200
    assert captured["occurred_at"] is None


def test_save_memory_null_occurred_at_400():
    response = client.post("/save_memory", json={"content": "distilled text", "occurred_at": None})
    assert response.status_code == 400
    assert "error" in response.json()


def test_save_memory_malformed_occurred_at_400(monkeypatch):
    async def fake_save_note(
        content, kind="note", tags=None, supersedes=None, namespace="default", occurred_at=None
    ):
        raise ValueError(f"invalid ISO 8601 timestamp: {occurred_at!r}")

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory", json={"content": "distilled text", "occurred_at": "nonsense"}
    )
    assert response.status_code == 400
    assert "invalid ISO 8601 timestamp" in response.json()["error"]


def test_save_memory_future_occurred_at_400(monkeypatch):
    async def fake_save_note(
        content, kind="note", tags=None, supersedes=None, namespace="default", occurred_at=None
    ):
        raise ValueError("occurred_at must not be in the future")

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory", json={"content": "distilled text", "occurred_at": "2999-01-01"}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "occurred_at must not be in the future"


def test_save_memory_episode_kind_delegates_to_save_note(monkeypatch):
    captured = {}

    async def fake_save_note(
        content, kind="note", tags=None, supersedes=None, namespace="default", occurred_at=None
    ):
        captured["kind"] = kind
        return {"id": "note:x", "kind": kind, "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory", json={"content": "caught up with the team today", "kind": "episode"}
    )
    assert response.status_code == 200
    assert captured["kind"] == "episode"
    assert response.json()["kind"] == "episode"


# ---- MCP: save_memory posts occurred_at in body ----------------------------


def _patch_client(monkeypatch, handler):
    import memory_base.serve.mcp_server as mcp_server

    def fake_client():
        return httpx.AsyncClient(
            base_url=mcp_server.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(mcp_server, "_client", fake_client)


def test_mcp_save_memory_posts_occurred_at_in_body(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "note:eeeeeeeeeeeeeeee",
                "kind": "episode",
                "stored": True,
                "superseded": None,
                "similar": [],
            },
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(
        save_memory("caught up with the team today", kind="episode", occurred_at="2023-06-01")
    )
    assert captured["json"]["occurred_at"] == "2023-06-01"
    assert captured["json"]["kind"] == "episode"


def test_mcp_save_memory_omits_occurred_at_when_not_given(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "note:ffffffffffffffff",
                "kind": "note",
                "stored": True,
                "superseded": None,
                "similar": [],
            },
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(save_memory("plain note"))
    assert "occurred_at" not in captured["json"]
