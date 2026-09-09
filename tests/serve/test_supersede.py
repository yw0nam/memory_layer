"""Contract tests for save_memory's supersedes + similar hints (red-first).

``save_note``/``save_memory`` take an
optional ``supersedes: <note id>`` argument (MCP tool, REST body, and
``save_note`` itself). Pinned contract:

- Response shape gains two keys: ``"superseded"`` (the id of the note that
  was archived because of this save, or ``None``) and ``"similar"`` (up to
  3 ``{id, score, text}`` hints for existing active notes whose embedding
  is close to the new content). Full shape:
  ``{"id", "kind", "stored", "superseded", "similar"}``.
- An unknown ``supersedes`` id raises ``ValueError`` with the message
  ``"unknown supersedes id: <id>"`` (pinned exactly); REST turns that into
  a 400 with ``{"error": "unknown supersedes id: <id>"}``.
- The MCP ``save_memory`` tool posts ``supersedes`` in the JSON body
  (``None`` when not given); the tool list is unaffected (still exactly
  {search, search_code, search_memory, save_memory, ingest_document}).

Pure/unit sections use no DB/network (REST route delegates to a monkeypatched
``api.save_note``; MCP proxy uses ``httpx.MockTransport`` as in
tests/test_mcp_proxy.py). Integration section (marked ``integration``,
skipped when the DB is unreachable) exercises the real stack via the
``rest_in_process``/direct REST client against Postgres + vLLM, cleaning up
every row it inserts.

Collection fails today: ``save_note``/``save_memory`` have no ``supersedes``
parameter and the response shape lacks ``superseded``/``similar``.
"""

from __future__ import annotations

import asyncio
import json
import time

import asyncpg
import httpx
import pytest
from starlette.testclient import TestClient

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api, mcp_server
from memory_base.serve.notes import build_note_row, save_note

NOW = 1_700_000_000.0


@pytest.fixture()
def client():
    with TestClient(api.app, headers={"X-API-Key": "test-key"}) as c:
        yield c


# ---- pure: response shape ---------------------------------------------------


def test_save_memory_response_shape_pins_superseded_and_similar(monkeypatch, client):
    async def fake_save_note(
        content,
        kind="note",
        tags=None,
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
    ):
        return {
            "id": "note:aaaaaaaaaaaaaaaa",
            "kind": "note",
            "stored": True,
            "superseded": None,
            "similar": [],
        }

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory", json={"author": "natsume", "content": "some distilled content"}
    )
    assert response.status_code == 200
    assert set(response.json()) == {"id", "kind", "stored", "superseded", "similar"}


# ---- REST: forwards supersedes, 400s on ValueError -------------------------


def test_save_memory_forwards_supersedes_to_save_note(monkeypatch, client):
    captured = {}

    async def fake_save_note(
        content,
        kind="note",
        tags=None,
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
    ):
        captured["supersedes"] = supersedes
        return {
            "id": "note:bbbbbbbbbbbbbbbb",
            "kind": "note",
            "stored": True,
            "superseded": supersedes,
            "similar": [],
        }

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={"author": "natsume", "content": "new content", "supersedes": "note:old0000000000"},
    )
    assert response.status_code == 200
    assert captured["supersedes"] == "note:old0000000000"
    assert response.json()["superseded"] == "note:old0000000000"


def test_save_memory_absent_supersedes_forwards_none(monkeypatch, client):
    captured = {}

    async def fake_save_note(
        content,
        kind="note",
        tags=None,
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
    ):
        captured["supersedes"] = supersedes
        return {
            "id": "note:cccccccccccccccc",
            "kind": "note",
            "stored": True,
            "superseded": None,
            "similar": [],
        }

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post("/save_memory", json={"author": "natsume", "content": "new content"})
    assert response.status_code == 200
    assert captured["supersedes"] is None
    assert response.json()["superseded"] is None


def test_save_memory_unknown_supersedes_id_400(monkeypatch, client):
    async def fake_save_note(
        content,
        kind="note",
        tags=None,
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
    ):
        raise ValueError(f"unknown supersedes id: {supersedes}")

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={"author": "natsume", "content": "new content", "supersedes": "note:missing00000000"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "unknown supersedes id: note:missing00000000"


# ---- save_note: the supersede UPDATE stamps archived_by --------------------


class FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(self):
        self.updates: list[tuple] = []

    def transaction(self):
        return FakeTransaction()

    async def fetchval(self, query, *args):
        return True

    async def execute(self, query, *args):
        if "SET archived_at" in query:
            self.updates.append((query, args))
            return "UPDATE 1"
        return "INSERT 0 1"

    async def fetch(self, query, *args):
        return []


def _patch_note_deps(monkeypatch, conn):
    from contextlib import asynccontextmanager

    from memory_base.serve import notes

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def fake_embed_text(embedder, text):
        return "[0]"

    async def noop(conn):
        return None

    monkeypatch.setattr(notes.db, "acquire", acquire)
    monkeypatch.setattr(notes, "embed_text", fake_embed_text)
    monkeypatch.setattr(notes, "VllmEmbedder", lambda: None)
    monkeypatch.setattr(notes, "ensure_schema_once", noop)


def test_supersede_stamps_archived_by_with_the_new_notes_author(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    asyncio.run(save_note("new content", supersedes="note:old0000000000", author="natsume"))
    query, args = conn.updates[0]
    assert "jsonb_build_object('archived_by'" in query
    assert "natsume" in args


# ---- MCP proxy: posts supersedes, tool list unchanged ----------------------


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=mcp_server.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(mcp_server, "_client", fake_client)


def test_mcp_save_memory_posts_supersedes_in_body(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "note:dddddddddddddddd",
                "kind": "note",
                "stored": True,
                "superseded": "note:old0000000000",
                "similar": [],
            },
        )

    _patch_client(monkeypatch, handler)
    result = asyncio.run(
        mcp_server.save_memory(
            "new content", "natsume", kind="note", tags=None, supersedes="note:old0000000000"
        )
    )
    assert captured["json"] == {
        "content": "new content",
        "author": "natsume",
        "kind": "note",
        "tags": None,
        "supersedes": "note:old0000000000",
    }
    assert result["superseded"] == "note:old0000000000"


def test_mcp_save_memory_posts_supersedes_none_when_absent(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "note:eeeeeeeeeeeeeeee",
                "kind": "note",
                "stored": True,
                "superseded": None,
                "similar": [],
            },
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(mcp_server.save_memory("new content", "natsume"))
    assert captured["json"] == {
        "content": "new content",
        "author": "natsume",
        "kind": "note",
        "tags": None,
        "supersedes": None,
    }


def test_mcp_tool_list_unaffected_by_supersede():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(
            mcp_server.mcp._mcp_server
        ) as client_session:
            result = await client_session.list_tools()
            return {t.name for t in result.tools}

    names = asyncio.run(_run())
    assert names == {
        "search",
        "search_code",
        "search_memory",
        "save_memory",
        "query_table",
        "ingest_document",
        "ingest_repo",
        "remove_repo",
        "remove_document",
        "list_repos",
        "list_notes",
        "list_memory_duplicates",
        "archive_notes",
        "restore_notes",
        "delete_notes",
    }


# ---- integration: real DB + embedder ----------------------------------------


def _db_reachable() -> bool:
    async def _check() -> None:
        conn = await asyncpg.connect(db_url(), timeout=5)
        await conn.close()

    try:
        asyncio.run(_check())
        return True
    except Exception:
        return False


_DB = _db_reachable()
requires_db = pytest.mark.skipif(not _DB, reason="DB is not configured or not reachable")


async def _delete(note_id: str) -> None:
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', note_id)
    finally:
        await conn.close()


async def _fetch_archived_at(note_id: str):
    conn = await asyncpg.connect(db_url())
    try:
        return await conn.fetchval(
            f'SELECT archived_at FROM "{PG_SCHEMA}".memory_chunks WHERE id=$1', note_id
        )
    finally:
        await conn.close()


@pytest.mark.integration
@requires_db
def test_supersede_archives_old_note_and_stores_new_one(client):
    content_a = f"supersede integration pin A {NOW}: zzzsupersedepin unique marker one"
    content_b = f"supersede integration pin B {NOW}: zzzsupersedepin unique marker two"
    note_a = build_note_row(content_a, "note", None, NOW)["id"]
    note_b = build_note_row(content_b, "note", None, NOW)["id"]
    asyncio.run(_delete(note_a))
    asyncio.run(_delete(note_b))
    try:
        response_a = client.post("/save_memory", json={"author": "natsume", "content": content_a})
        assert response_a.status_code == 200
        assert response_a.json()["id"] == note_a

        response_b = client.post(
            "/save_memory", json={"author": "natsume", "content": content_b, "supersedes": note_a}
        )
        assert response_b.status_code == 200
        assert response_b.json()["id"] == note_b
        assert response_b.json()["superseded"] == note_a

        assert asyncio.run(_fetch_archived_at(note_a)) is not None
        assert asyncio.run(_fetch_archived_at(note_b)) is None
    finally:
        asyncio.run(_delete(note_a))
        asyncio.run(_delete(note_b))


@pytest.mark.integration
@requires_db
def test_supersede_unknown_id_400_over_rest(client):
    content = f"supersede integration pin C {NOW}: zzzsupersedepin unique marker three"
    response = client.post(
        "/save_memory",
        json={
            "author": "natsume",
            "content": content,
            "supersedes": "note:0000000000000000",
        },
    )
    assert response.status_code == 400
    assert "unknown supersedes id" in response.json()["error"]


@pytest.mark.integration
@requires_db
def test_similar_hints_include_near_identical_active_note(client):
    marker = f"zzzsimilarpin{int(NOW)}"
    content_b = f"similar-hint integration pin: {marker} a hard-won troubleshooting conclusion"
    content_c = f"similar-hint integration pin: {marker} a hard won troubleshooting conclusion!"
    note_b = build_note_row(content_b, "note", None, NOW)["id"]
    note_c = build_note_row(content_c, "note", None, NOW)["id"]
    asyncio.run(_delete(note_b))
    asyncio.run(_delete(note_c))
    try:
        asyncio.run(save_note(content_b))
        time.sleep(0.2)  # let the embedder-backed insert settle before querying similarity

        response_c = client.post("/save_memory", json={"author": "natsume", "content": content_c})
        assert response_c.status_code == 200
        similar_ids = [item["id"] for item in response_c.json()["similar"]]
        assert note_b in similar_ids
        for item in response_c.json()["similar"]:
            assert {"id", "score", "text"} <= set(item)
    finally:
        asyncio.run(_delete(note_b))
        asyncio.run(_delete(note_c))
