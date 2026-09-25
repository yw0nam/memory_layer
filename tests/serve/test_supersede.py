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
from memory_base.serve.notes import (
    ContentVerdict,
    SimilarNotesError,
    build_note_row,
    save_note,
)

NOW = 1_700_000_000.0


@pytest.fixture()
def client():
    with TestClient(api.app, headers={"X-API-Key": "test-key"}) as c:
        yield c


# ---- pure: response shape ---------------------------------------------------


def test_save_memory_response_shape_pins_superseded_and_similar(monkeypatch, client):
    async def fake_save_note(
        content,
        *,
        tags,
        kind="note",
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
        allow_similar=False,
        allow_restatement=False,
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
        *,
        tags,
        kind="note",
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
        allow_similar=False,
        allow_restatement=False,
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
        *,
        tags,
        kind="note",
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
        allow_similar=False,
        allow_restatement=False,
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
        *,
        tags,
        kind="note",
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
        allow_similar=False,
        allow_restatement=False,
    ):
        raise ValueError(f"unknown supersedes id: {supersedes}")

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={"author": "natsume", "content": "new content", "supersedes": "note:missing00000000"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "unknown supersedes id: note:missing00000000"


# ---- REST: the near-duplicate gate ------------------------------------------


def test_save_memory_similar_notes_error_409(monkeypatch, client):
    similar = [{"id": "note:neighbour000000", "score": 0.97, "text": "nearly the same content"}]

    async def fake_save_note(
        content,
        *,
        tags,
        kind="note",
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
        allow_similar=False,
        allow_restatement=False,
    ):
        raise SimilarNotesError(similar)

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post("/save_memory", json={"author": "natsume", "content": "new content"})
    assert response.status_code == 409
    body = response.json()
    assert "Refused: 1 active note(s)" in body["error"]
    assert "note:neighbour000000" in body["error"]
    assert body["similar"] == similar


def test_save_memory_non_bool_allow_similar_400(monkeypatch, client):
    async def fake_save_note(content, **kwargs):
        return {"id": "note:x", "kind": "note", "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={"author": "natsume", "content": "new content", "allow_similar": "yes"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "allow_similar must be a boolean"


def test_save_memory_forwards_allow_similar_to_save_note(monkeypatch, client):
    captured = {}

    async def fake_save_note(
        content,
        *,
        tags,
        kind="note",
        supersedes=None,
        namespace="default",
        occurred_at=None,
        author=None,
        allow_similar=False,
        allow_restatement=False,
    ):
        captured["allow_similar"] = allow_similar
        return {"id": "note:x", "kind": kind, "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={"author": "natsume", "content": "new content", "allow_similar": True},
    )
    assert response.status_code == 200
    assert captured["allow_similar"] is True
    response = client.post("/save_memory", json={"author": "natsume", "content": "new content"})
    assert response.status_code == 200
    assert captured["allow_similar"] is False


# ---- save_note: the supersede UPDATE stamps archived_by --------------------


class FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(self, similar_rows=None, insert_status="INSERT 0 1"):
        self.updates: list[tuple] = []
        self.similar_rows = similar_rows or []
        self.insert_status = insert_status
        self.inserts: list[tuple] = []

    def transaction(self):
        return FakeTransaction()

    async def fetchval(self, query, *args):
        return True

    async def execute(self, query, *args):
        if "SET archived_at" in query:
            self.updates.append((query, args))
            return "UPDATE 1"
        self.inserts.append((query, args))
        return self.insert_status

    async def fetch(self, query, *args):
        return self.similar_rows


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

    async def accepted_judge(content, kind):
        return ContentVerdict(accepted=True, reason="durable knowledge")

    monkeypatch.setattr(notes, "judge_note_content", accepted_judge)


def test_supersede_stamps_archived_by_with_the_new_notes_author(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    asyncio.run(
        save_note("new content", tags=["test"], supersedes="note:old0000000000", author="natsume")
    )
    query, args = conn.updates[0]
    assert "jsonb_build_object('archived_by'" in query
    assert "natsume" in args


# ---- save_note: the near-duplicate gate -------------------------------------


def _neighbour():
    return {"id": "note:neighbour000000", "score": 0.97, "text": "nearly the same content"}


def test_save_note_refused_next_to_near_identical_active_note(monkeypatch):
    neighbour = _neighbour()
    conn = FakeConnection(similar_rows=[neighbour])
    _patch_note_deps(monkeypatch, conn)
    with pytest.raises(SimilarNotesError) as exc_info:
        asyncio.run(save_note("new content", tags=["test"], author="natsume"))
    assert exc_info.value.similar == [neighbour]
    message = str(exc_info.value)
    assert neighbour["id"] in message
    assert "supersedes" in message
    assert "allow_similar" in message
    assert conn.updates == []


def test_save_note_with_supersedes_of_the_neighbour_passes(monkeypatch):
    neighbour = _neighbour()
    conn = FakeConnection(similar_rows=[neighbour])
    _patch_note_deps(monkeypatch, conn)
    result = asyncio.run(
        save_note("new content", tags=["test"], supersedes=neighbour["id"], author="natsume")
    )
    assert result["stored"] is True
    assert result["superseded"] == neighbour["id"]
    assert result["similar"] == []
    assert len(conn.updates) == 1


def test_save_note_with_allow_similar_records_similar_ack(monkeypatch):
    neighbour = _neighbour()
    conn = FakeConnection(similar_rows=[neighbour])
    _patch_note_deps(monkeypatch, conn)
    result = asyncio.run(
        save_note("new content", tags=["test"], allow_similar=True, author="natsume")
    )
    assert result["stored"] is True
    assert result["similar"] == [neighbour]
    _, args = conn.inserts[0]
    metadata = json.loads(args[11])
    assert metadata["similar_ack"] == [neighbour["id"]]


def test_save_note_allow_similar_with_supersedes_neighbour_stamps_nothing(monkeypatch):
    neighbour = _neighbour()
    conn = FakeConnection(similar_rows=[neighbour])
    _patch_note_deps(monkeypatch, conn)
    result = asyncio.run(
        save_note(
            "new content",
            tags=["test"],
            supersedes=neighbour["id"],
            allow_similar=True,
            author="natsume",
        )
    )
    assert result["stored"] is True
    _, args = conn.inserts[0]
    metadata = json.loads(args[11])
    assert "similar_ack" not in metadata
    assert result["similar"] == []


def test_save_note_not_stored_is_never_gated(monkeypatch):
    neighbour = _neighbour()
    conn = FakeConnection(similar_rows=[neighbour], insert_status="INSERT 0 0")
    _patch_note_deps(monkeypatch, conn)
    result = asyncio.run(save_note("new content", tags=["test"], author="natsume"))
    assert result["stored"] is False


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
        "allow_similar": False,
        "allow_restatement": False,
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
    asyncio.run(mcp_server.save_memory("new content", "natsume", tags=["test"]))
    assert captured["json"] == {
        "content": "new content",
        "author": "natsume",
        "kind": "note",
        "tags": ["test"],
        "supersedes": None,
        "allow_similar": False,
        "allow_restatement": False,
    }


def test_mcp_save_memory_posts_allow_similar_true_when_passed(monkeypatch):
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
    asyncio.run(mcp_server.save_memory("new content", "natsume", tags=["test"], allow_similar=True))
    assert captured["json"]["allow_similar"] is True


def test_mcp_save_memory_409_surfaces_backend_message(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "Refused: 1 active note(s) say the same thing"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError, match=r"Refused: 1 active note\(s\) say the same thing"):
        asyncio.run(mcp_server.save_memory("new content", "natsume", tags=["test"]))


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
        "send_message",
        "list_messages",
        "claim_message",
        "cancel_message",
    }


# ---- integration: real DB + embedder ----------------------------------------


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


async def _fetch_similar_ack(note_id: str):
    conn = await asyncpg.connect(db_url())
    try:
        return await conn.fetchval(
            f"SELECT metadata->'similar_ack' FROM \"{PG_SCHEMA}\".memory_chunks WHERE id=$1",
            note_id,
        )
    finally:
        await conn.close()


@pytest.mark.integration
def test_supersede_archives_old_note_and_stores_new_one(client):
    content_a = f"supersede integration pin A {NOW}: zzzsupersedepin unique marker one"
    content_b = f"supersede integration pin B {NOW}: zzzsupersedepin unique marker two"
    note_a = build_note_row(content_a, "note", ["test"], NOW)["id"]
    note_b = build_note_row(content_b, "note", ["test"], NOW)["id"]
    asyncio.run(_delete(note_a))
    asyncio.run(_delete(note_b))
    try:
        response_a = client.post(
            "/save_memory", json={"author": "natsume", "content": content_a, "tags": ["test"]}
        )
        assert response_a.status_code == 200
        assert response_a.json()["id"] == note_a

        response_b = client.post(
            "/save_memory",
            json={
                "author": "natsume",
                "content": content_b,
                "tags": ["test"],
                "supersedes": note_a,
            },
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
def test_supersede_unknown_id_400_over_rest(client):
    content = f"supersede integration pin C {NOW}: zzzsupersedepin unique marker three"
    response = client.post(
        "/save_memory",
        json={
            "author": "natsume",
            "content": content,
            "tags": ["test"],
            "supersedes": "note:0000000000000000",
        },
    )
    assert response.status_code == 400
    assert "unknown supersedes id" in response.json()["error"]


@pytest.mark.integration
def test_similar_gate_refuses_then_resolves_by_supersede_or_ack(client):
    marker = f"zzzsimilarpin{int(NOW)}"
    content_b = f"similar-hint integration pin: {marker} a hard-won troubleshooting conclusion"
    content_c = f"similar-hint integration pin: {marker} a hard won troubleshooting conclusion!"
    note_b = build_note_row(content_b, "note", ["test"], NOW)["id"]
    note_c = build_note_row(content_c, "note", ["test"], NOW)["id"]
    asyncio.run(_delete(note_b))
    asyncio.run(_delete(note_c))
    try:
        asyncio.run(save_note(content_b, tags=["test"]))
        time.sleep(0.2)  # let the embedder-backed insert settle before querying similarity

        response_c = client.post(
            "/save_memory", json={"author": "natsume", "content": content_c, "tags": ["test"]}
        )
        assert response_c.status_code == 409
        body = response_c.json()
        assert note_b in body["error"]
        assert note_b in [item["id"] for item in body["similar"]]
        for item in body["similar"]:
            assert {"id", "score", "text"} <= set(item)

        response_ack = client.post(
            "/save_memory",
            json={
                "author": "natsume",
                "content": content_c,
                "tags": ["test"],
                "allow_similar": True,
            },
        )
        assert response_ack.status_code == 200
        assert response_ack.json()["id"] == note_c
        assert json.loads(asyncio.run(_fetch_similar_ack(note_c))) == [note_b]

        response_sup = client.post(
            "/save_memory",
            json={
                "author": "natsume",
                "content": content_c,
                "tags": ["test"],
                "supersedes": note_b,
            },
        )
        assert response_sup.status_code == 200
        assert asyncio.run(_fetch_archived_at(note_b)) is not None
    finally:
        asyncio.run(_delete(note_b))
        asyncio.run(_delete(note_c))
