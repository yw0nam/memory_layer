"""Contract tests for the personal and work note kinds (red-first).

Notes are saved through one MCP tool, ``save_memory``, whose required ``kind``
(``personal`` / ``work``) labels the note for search and listing and never decides
whether it is stored. ``POST /save_memory`` requires ``kind``; reads accept only these
kinds.

Unit sections follow tests/serve/test_save_without_gate.py's FakeConnection pattern
(no DB, no network). Integration sections run the real stack in-process.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager

import asyncpg
import httpx
import pytest
from starlette.testclient import TestClient

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api, mcp_server
from memory_base.serve.common import rest_client
from memory_base.serve.mcp_server import SERVER_INSTRUCTIONS
from memory_base.serve.notes import curation, store, tools
from memory_base.serve.notes.store import (
    build_note_row,
    save_note,
)

NOW = 1_700_000_000.0
KIND_ERROR = "kind must be one of ('personal', 'work')"
DESCRIPTION_LIMIT = 2048  # a client that shows a tool description or the instructions cuts here

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


# ---- save_note: the kind contract (no DB/network) -----------------------------


class FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(self, insert_status: str = "INSERT 0 1", stored_kind: str = "work"):
        self.insert_status = insert_status
        self.stored_kind = stored_kind
        self.embeds: list[str] = []
        self.insert_args: tuple | None = None
        self.kind_reads = 0

    def transaction(self):
        return FakeTransaction()

    async def fetchval(self, query, *args):
        if "namespaces" in query:
            return True
        if "chunk_kind" in query:
            self.kind_reads += 1
            return self.stored_kind
        return True

    async def execute(self, query, *args):
        if "INSERT INTO" in query:
            self.insert_args = args
            return self.insert_status
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
        conn.embeds.append(text)
        return "[0]"

    monkeypatch.setattr(store.db, "acquire", acquire)
    monkeypatch.setattr(store, "embed_text", fake_embed_text)
    monkeypatch.setattr(store, "VllmEmbedder", lambda: None)
    monkeypatch.setattr(store, "ensure_schema_once", _noop)


@pytest.mark.parametrize("kind", [None, True, "note"])
def test_save_note_rejects_a_malformed_kind(monkeypatch, kind):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    with pytest.raises(ValueError, match=re.escape(KIND_ERROR)):
        asyncio.run(save_note("distilled content", tags=["test"], kind=kind))
    assert conn.embeds == []
    assert conn.insert_args is None


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_either_kind_is_stored_with_its_label(monkeypatch, kind):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    result = asyncio.run(save_note("distilled content", tags=["test"], kind=kind))
    assert conn.embeds == ["distilled content"]
    assert result["stored"] is True
    assert result["kind"] == kind


# ---- REST: POST /save_memory --------------------------------------------------


def test_content_errors_win_over_a_missing_kind():
    response = client.post("/save_memory", json={"author": "natsume", "content": ""})
    assert response.status_code == 400
    assert response.json()["error"] == "content must not be empty"


def test_duplicates_route_rejects_an_unknown_kind(monkeypatch):
    calls = []

    async def fake_find_duplicates(threshold, kind, limit, namespaces=None):
        calls.append(kind)
        return []

    monkeypatch.setattr(curation, "find_duplicates", fake_find_duplicates)
    response = client.get("/admin/duplicates", params={"kind": "note"})
    assert response.status_code == 400
    assert response.json()["error"] == KIND_ERROR
    assert calls == []
    assert client.get("/admin/duplicates", params={"kind": "work"}).status_code == 200
    assert calls == ["work"]


# ---- MCP: the save tool ----------------------------------------------------------


def _tools():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(
            mcp_server.mcp._mcp_server
        ) as session:
            result = await session.list_tools()
            return {t.name: t for t in result.tools}

    return asyncio.run(_run())


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(rest_client, "client", fake_client)


def test_save_memory_schema_requires_a_kind_of_two_values():
    schema = _tools()["save_memory"].inputSchema
    assert {"content", "author", "tags", "kind"} <= set(schema["required"])
    assert schema["properties"]["kind"]["enum"] == ["personal", "work"]
    assert "track" not in schema["properties"]
    assert schema["properties"]["allow_similar"]["type"] == "boolean"
    assert "allow_similar" not in schema["required"]
    assert "allow_restatement" not in schema["properties"]


def test_save_memory_parameters_carry_their_notes_in_the_schema():
    properties = _tools()["save_memory"].inputSchema["properties"]
    for name in (
        "kind",
        "tags",
        "author",
        "supersedes",
        "allow_similar",
        "occurred_at",
        "namespace",
    ):
        assert properties[name].get("description"), name
    assert "never decides whether the note is stored" in properties["kind"]["description"]
    assert "first tag names the subject" in properties["tags"]["description"]
    assert "author allowlist" in properties["author"]["description"]


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_save_memory_posts_the_kind_it_is_given(monkeypatch, kind):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "note:abc", "kind": kind, "stored": True})

    _patch_client(monkeypatch, handler)
    result = asyncio.run(
        tools.save_memory("distilled content", "natsume", tags=["infra"], kind=kind)
    )
    assert captured["path"] == "/save_memory"
    assert captured["json"] == {
        "content": "distilled content",
        "author": "natsume",
        "kind": kind,
        "tags": ["infra"],
        "supersedes": None,
        "allow_similar": False,
    }
    assert result == {"id": "note:abc", "kind": kind, "stored": True}


def test_save_memory_description_states_both_bars_and_the_label():
    description = " ".join(_tools()["save_memory"].description.split())
    for anchor in (
        "`kind` labels the note for a later search",
        "The label never decides whether a note is stored",
        "a moment between you, or something you made or gave them",
        "a decision and its reason, a reproduced bug and its fix",
        "a decision without its reason is not worth keeping",
        "Do NOT save progress or next steps of the current session",
        "send_message",
        "English",
        "search_memory the same subject",
        "archive them with archive_notes",
        "The server stores a note once it passes the validation, credential, and near-duplicate "
        "checks",
    ):
        assert anchor in description
    for stale in (
        "save_personal_memory",
        "save_work_memory",
        "the other save tool",
        "rewrite it once",
    ):
        assert stale not in description


def test_save_tool_summary_fits_a_tool_catalog():
    # A deferring client shows the first sentence clipped to 60 characters.
    description = " ".join(_tools()["save_memory"].description.split())
    first = re.match(r"(.+?[.!?])(?=\s|$)", description).group(1)
    assert first == "Remember the user and their work for later sessions."
    assert len(first) <= 60


def test_every_tool_description_fits_what_a_client_shows():
    lengths = {name: len(tool.description or "") for name, tool in _tools().items()}
    assert max(lengths.values()) <= DESCRIPTION_LIMIT, lengths


def test_the_write_policy_fits_what_a_client_shows_of_the_instructions():
    assert (
        SERVER_INSTRUCTIONS.index(mcp_server._WRITE_POLICY) + len(mcp_server._WRITE_POLICY)
        <= DESCRIPTION_LIMIT
    )


def test_server_instructions_name_the_save_tool_and_the_kind_as_a_label():
    flat = " ".join(SERVER_INSTRUCTIONS.split())
    for anchor in (
        "save_memory",
        "send_message",
        "Write rarely.",
        "a moment they shared with you, even during work",
        "The kind only labels a note and never decides whether it is stored.",
    ):
        assert anchor in flat
    for stale in (
        "save_personal_memory",
        "save_work_memory",
        "the other save tool",
        "at most once",
    ):
        assert stale not in flat


def test_server_instructions_carry_no_per_job_criteria():
    for stale in ("a reproduced bug", "non-obvious environment fact", "allow_restatement"):
        assert stale not in SERVER_INSTRUCTIONS


# ---- integration: real DB + embedder --------------------------------------------


async def _rows(note_id: str):
    conn = await asyncpg.connect(db_url())
    try:
        return await conn.fetch(
            f'SELECT chunk_kind, archived_at, metadata FROM "{PG_SCHEMA}".memory_chunks '
            "WHERE id=$1",
            note_id,
        )
    finally:
        await conn.close()


async def _delete(*note_ids: str) -> None:
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE id = ANY($1::text[])', list(note_ids)
        )
    finally:
        await conn.close()


def _marker() -> str:
    return f"zzzkind{uuid.uuid4().hex[:10]}"


@pytest.mark.integration
def test_cross_kind_duplicate_is_a_no_op_that_keeps_the_first_kind(rest_in_process):
    content = f"note kinds integration pin {_marker()}: identical content through both tools"
    note_id = build_note_row(content, "work", ["test"], NOW)["id"]
    asyncio.run(_delete(note_id))
    try:
        first = asyncio.run(tools.save_memory(content, "natsume", tags=["test"], kind="work"))
        second = asyncio.run(tools.save_memory(content, "natsume", tags=["test"], kind="personal"))
        assert first["stored"] is True
        assert second["stored"] is False
        assert second["id"] == first["id"] == note_id
        assert second["kind"] == "work"
        rows = asyncio.run(_rows(note_id))
        assert [row["chunk_kind"] for row in rows] == ["work"]
    finally:
        asyncio.run(_delete(note_id))


@pytest.mark.integration
def test_search_and_listing_filter_by_the_new_kinds():
    marker = _marker()
    personal = f"note kinds filter pin {marker}: the user keeps two cats named Kiwi and Pear"
    work = f"note kinds filter pin {marker}: the nightly job fails when the disk is full"
    ids = [build_note_row(c, "work", ["test"], NOW)["id"] for c in (personal, work)]
    asyncio.run(_delete(*ids))
    try:
        with TestClient(api.app, headers={"X-API-Key": "test-key"}) as c:
            for content, kind in ((personal, "personal"), (work, "work")):
                response = c.post(
                    "/save_memory",
                    json={
                        "author": "natsume",
                        "content": content,
                        "kind": kind,
                        "tags": [marker],
                        "allow_similar": True,
                    },
                )
                assert response.status_code == 200, response.json()
            found = c.post(
                "/search",
                json={"query": marker, "source": "memory", "kind": "personal"},
            )
            assert found.status_code == 200
            assert [hit["id"] for hit in found.json()] == [ids[0]]
            listed = c.get("/notes", params={"kind": "work", "tags": marker})
            assert listed.status_code == 200
            assert [row["id"] for row in listed.json()] == [ids[1]]
            for bad in (
                c.post("/search", json={"query": marker, "source": "memory", "kind": "note"}),
                c.get("/notes", params={"kind": "note"}),
            ):
                assert bad.status_code == 400
    finally:
        asyncio.run(_delete(*ids))


@pytest.mark.integration
def test_cross_kind_supersede_archives_the_other_kinds_note(rest_in_process):
    marker = _marker()
    old = f"note kinds supersede pin {marker}: the deploy window is on Tuesdays"
    new = f"note kinds supersede pin {marker}: the deploy window is on Thursdays (Tuesdays before)"
    old_id = build_note_row(old, "work", ["test"], NOW)["id"]
    new_id = build_note_row(new, "work", ["test"], NOW)["id"]
    asyncio.run(_delete(old_id, new_id))
    try:
        saved_old = asyncio.run(tools.save_memory(old, "natsume", tags=["test"], kind="work"))
        assert saved_old["id"] == old_id
        saved_new = asyncio.run(
            tools.save_memory(new, "natsume", tags=["test"], kind="personal", supersedes=old_id)
        )
        assert saved_new["id"] == new_id
        assert saved_new["superseded"] == old_id
        old_row = asyncio.run(_rows(old_id))[0]
        new_row = asyncio.run(_rows(new_id))[0]
        assert old_row["archived_at"] is not None
        assert json.loads(new_row["metadata"])["supersedes"] == old_id
        assert new_row["chunk_kind"] == "personal"
    finally:
        asyncio.run(_delete(old_id, new_id))
