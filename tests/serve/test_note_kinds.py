"""Contract tests for the personal and work note kinds (red-first).

Notes are saved through two MCP tools, ``save_personal_memory`` and
``save_work_memory``; each fixes the stored kind (``personal`` / ``work``) and
is judged by the content gate with the prompt of its kind. ``POST /save_memory``
requires ``kind``; reads accept only the new kinds.

Unit sections follow tests/serve/test_content_gate.py's FakeConnection pattern
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
from memory_base.serve import admin, api, mcp_server, notes
from memory_base.serve.mcp_server import SERVER_INSTRUCTIONS
from memory_base.serve.notes import (
    JUDGE_PROMPTS,
    NOTE_KINDS,
    ContentVerdict,
    LowSignalNoteError,
    build_note_row,
    judge_note_content,
    save_note,
)
from test_content_gate import _assert_refusal_text

NOW = 1_700_000_000.0
KIND_ERROR = "kind must be one of ('personal', 'work')"
SAVE_TOOLS = [("save_personal_memory", "personal"), ("save_work_memory", "work")]

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


def _patch_note_deps(monkeypatch, conn, judge=None):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def fake_embed_text(embedder, text):
        conn.embeds.append(text)
        return "[0]"

    async def accepted_judge(content, kind):
        return ContentVerdict(accepted=True, reason="durable knowledge")

    monkeypatch.setattr(notes.db, "acquire", acquire)
    monkeypatch.setattr(notes, "embed_text", fake_embed_text)
    monkeypatch.setattr(notes, "VllmEmbedder", lambda: None)
    monkeypatch.setattr(notes, "ensure_schema_once", _noop)
    monkeypatch.setattr(notes, "judge_note_content", judge or accepted_judge)


def test_save_note_requires_a_kind():
    with pytest.raises(TypeError):
        asyncio.run(save_note("distilled content", tags=["test"]))
    with pytest.raises(ValueError, match=re.escape(KIND_ERROR)):
        asyncio.run(save_note("distilled content", tags=["test"], kind=None))


@pytest.mark.parametrize("kind", [True, ["work"], {"k": 1}, "note", "decision", "episode", "other"])
def test_save_note_rejects_a_malformed_kind(monkeypatch, kind):
    conn = FakeConnection()

    async def never_called(content, kind):
        raise AssertionError("the gate must not run for a malformed kind")

    _patch_note_deps(monkeypatch, conn, judge=never_called)
    with pytest.raises(ValueError, match=re.escape(KIND_ERROR)):
        asyncio.run(save_note("distilled content", tags=["test"], kind=kind))
    assert conn.embeds == []
    assert conn.insert_args is None


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_the_gate_receives_the_notes_kind(monkeypatch, kind):
    conn = FakeConnection()
    seen = []

    async def recording_judge(content, kind):
        seen.append((content, kind))
        return ContentVerdict(accepted=True, reason="durable knowledge")

    _patch_note_deps(monkeypatch, conn, judge=recording_judge)
    result = asyncio.run(save_note("distilled content", tags=["test"], kind=kind))
    assert seen == [("distilled content", kind)]
    assert result["stored"] is True
    assert result["kind"] == kind


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_judge_uses_the_prompt_of_its_kind(monkeypatch, kind):
    captured = {}

    async def fake_chat_json(messages, schema, *, timeout):
        captured["messages"] = messages
        captured["schema"] = schema
        captured["timeout"] = timeout
        return {"accepted": True, "reason": "ok"}

    monkeypatch.setattr(notes, "chat_json", fake_chat_json)
    verdict = asyncio.run(judge_note_content("distilled content", kind))
    assert verdict == ContentVerdict(accepted=True, reason="ok")
    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    assert captured["messages"][0]["content"] == JUDGE_PROMPTS[kind]
    assert captured["messages"][1]["content"] == "distilled content"
    assert captured["schema"]["required"] == ["accepted", "reason"]
    assert captured["timeout"] == notes.NOTE_GATE_TIMEOUT_SECONDS


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_judge_failure_fails_open_for_either_kind(monkeypatch, kind):
    conn = FakeConnection()

    async def failing_judge(content, kind):
        raise TimeoutError("chat timed out")

    _patch_note_deps(monkeypatch, conn, judge=failing_judge)
    result = asyncio.run(save_note("distilled content", tags=["test"], kind=kind))
    assert result["stored"] is True
    assert json.loads(conn.insert_args[10])["content_gate"] == "unavailable"


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_refusal_text_names_the_other_tool(monkeypatch, kind):
    conn = FakeConnection()

    async def refusing_judge(content, kind):
        return ContentVerdict(accepted=False, reason="not for this gate")

    _patch_note_deps(monkeypatch, conn, judge=refusing_judge)
    with pytest.raises(LowSignalNoteError) as exc_info:
        asyncio.run(save_note("distilled content", tags=["test"], kind=kind))
    assert exc_info.value.reason == "not for this gate"
    assert exc_info.value.kind == kind
    _assert_refusal_text(str(exc_info.value), "not for this gate", kind)
    assert conn.embeds == []
    assert conn.insert_args is None


def test_judge_prompts_state_their_contracts():
    assert NOTE_KINDS == ("personal", "work")
    assert set(JUDGE_PROMPTS) == set(NOTE_KINDS)
    flat = {kind: " ".join(prompt.split()) for kind, prompt in JUDGE_PROMPTS.items()}
    for text in flat.values():
        assert text.endswith("State the reason in one sentence.")
        assert "must be split into" in text
        assert "is not a mix" in text
        for anchor in ("PR", "namespace", "file does", "session that produced"):
            assert anchor not in text
        assert not re.search(r"\bcommit\b", text)
    for anchor in ("Judge generously.", "A note's moment never refuses it", "technical or not"):
        assert anchor in flat["personal"]
    for anchor in (
        "Judge strictly.",
        "A copy that carries what its source does not state still fails",
        "the failure and its fix, stated outright, pass",
    ):
        assert anchor in flat["work"]


def test_identical_content_through_the_other_tool_returns_the_stored_kind(monkeypatch):
    conn = FakeConnection(insert_status="INSERT 0 0", stored_kind="work")
    judged = []

    async def recording_judge(content, kind):
        judged.append(kind)
        return ContentVerdict(accepted=True, reason="durable knowledge")

    _patch_note_deps(monkeypatch, conn, judge=recording_judge)
    result = asyncio.run(save_note("distilled content", tags=["test"], kind="personal"))
    assert result["stored"] is False
    assert result["kind"] == "work"
    assert judged == ["personal"]


def test_identical_content_still_passes_the_chosen_tools_gate(monkeypatch):
    conn = FakeConnection(insert_status="INSERT 0 0")

    async def refusing_judge(content, kind):
        return ContentVerdict(accepted=False, reason="not for this gate")

    _patch_note_deps(monkeypatch, conn, judge=refusing_judge)
    with pytest.raises(LowSignalNoteError):
        asyncio.run(save_note("distilled content", tags=["test"], kind="personal"))
    assert conn.insert_args is None
    assert conn.kind_reads == 0


def test_the_kind_does_not_enter_the_note_id():
    personal = build_note_row("distilled content", "personal", ["test"], NOW)
    work = build_note_row("distilled content", "work", ["test"], NOW)
    assert personal["id"] == work["id"]


# ---- REST: POST /save_memory --------------------------------------------------


def test_save_memory_route_requires_kind():
    response = client.post(
        "/save_memory", json={"author": "natsume", "content": "valid content", "tags": ["test"]}
    )
    assert response.status_code == 400
    assert response.json()["error"] == KIND_ERROR


@pytest.mark.parametrize("kind", [None, True, [], {}, "note", "other"])
def test_save_memory_route_rejects_a_malformed_kind(kind):
    response = client.post(
        "/save_memory",
        json={"author": "natsume", "content": "valid content", "tags": ["test"], "kind": kind},
    )
    assert response.status_code == 400
    assert response.json()["error"] == KIND_ERROR


def test_content_errors_win_over_a_missing_kind():
    response = client.post("/save_memory", json={"author": "natsume", "content": ""})
    assert response.status_code == 400
    assert response.json()["error"] == "content must not be empty"


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_save_memory_route_forwards_kind_to_save_note(monkeypatch, kind):
    captured = {}

    async def fake_save_note(
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
        captured["kind"] = kind
        return {"id": "note:x", "kind": kind, "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={"author": "natsume", "content": "distilled text", "tags": ["test"], "kind": kind},
    )
    assert response.status_code == 200
    assert captured["kind"] == kind


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_low_signal_refusal_maps_to_409_with_the_kinds_text(monkeypatch, kind):
    async def fake_save_note(content, **kwargs):
        raise LowSignalNoteError("not for this gate", kind)

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={"author": "natsume", "content": "distilled text", "tags": ["test"], "kind": kind},
    )
    assert response.status_code == 409
    body = response.json()
    assert body["reason"] == "not for this gate"
    _assert_refusal_text(body["error"], "not for this gate", kind)


def test_duplicates_route_rejects_an_unknown_kind(monkeypatch):
    calls = []

    async def fake_find_duplicates(threshold, kind, limit, namespaces=None):
        calls.append(kind)
        return []

    monkeypatch.setattr(admin, "find_duplicates", fake_find_duplicates)
    response = client.get("/admin/duplicates", params={"kind": "note"})
    assert response.status_code == 400
    assert response.json()["error"] == KIND_ERROR
    assert calls == []
    assert client.get("/admin/duplicates", params={"kind": "work"}).status_code == 200
    assert calls == ["work"]


# ---- MCP: the two save tools ----------------------------------------------------


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
            base_url=mcp_server.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(mcp_server, "_client", fake_client)


def test_tool_list_offers_the_two_save_tools_and_no_save_memory():
    names = set(_tools())
    assert {"save_personal_memory", "save_work_memory"} <= names
    assert "save_memory" not in names


@pytest.mark.parametrize("tool", ["save_personal_memory", "save_work_memory"])
def test_save_tool_schemas_exclude_kind_and_track(tool):
    schema = _tools()[tool].inputSchema
    assert "kind" not in schema["properties"]
    assert "track" not in schema["properties"]
    assert {"content", "author", "tags"} <= set(schema["required"])
    assert schema["properties"]["allow_similar"]["type"] == "boolean"
    assert "allow_similar" not in schema["required"]
    assert "allow_restatement" not in schema["properties"]


@pytest.mark.parametrize(("tool", "kind"), SAVE_TOOLS)
def test_each_save_tool_posts_its_fixed_kind(monkeypatch, tool, kind):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "note:abc", "kind": kind, "stored": True})

    _patch_client(monkeypatch, handler)
    result = asyncio.run(getattr(mcp_server, tool)("distilled content", "natsume", tags=["infra"]))
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


@pytest.mark.parametrize(("tool", "kind"), SAVE_TOOLS)
def test_each_save_tool_surfaces_the_gate_refusal(monkeypatch, tool, kind):
    refusal = LowSignalNoteError("not for this gate", kind)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": str(refusal), "reason": refusal.reason})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(getattr(mcp_server, tool)("distilled content", "natsume", tags=["test"]))
    assert str(exc_info.value) == str(refusal)
    _assert_refusal_text(str(exc_info.value), "not for this gate", kind)


def test_save_tool_descriptions_state_their_criteria():
    tools = _tools()
    personal = tools["save_personal_memory"].description
    work = tools["save_work_memory"].description
    for anchor in (
        "a moment between you, even one during work,",
        "Do NOT save work knowledge here",
        "technical or not",
        "save_work_memory",
        "send_message",
        "English",
        "search_memory the same subject",
        "archive them with archive_notes",
        "once in total",
    ):
        assert anchor in personal
    for anchor in (
        "Record work knowledge",
        "The bar is strict",
        "Do NOT save",
        "save_personal_memory",
        "send_message",
        "English",
        "search_memory the same subject",
        "archive them with archive_notes",
        "once in total",
    ):
        assert anchor in work
    for description in (personal, work):
        assert "split, each part saved with its own tool" in " ".join(description.split())


def test_save_tool_summaries_fit_a_tool_catalog():
    # A deferring client shows the first sentence clipped to 60 characters.
    tools = _tools()
    first = {
        name: re.match(r"(.+?[.!?])(?=\s|$)", " ".join(tools[name].description.split())).group(1)
        for name in ("save_personal_memory", "save_work_memory")
    }
    assert first == {
        "save_personal_memory": "Remember the user: their life, their day, moments with you.",
        "save_work_memory": "Record work knowledge no code, commit, or tracker holds.",
    }
    assert all(len(sentence) <= 60 for sentence in first.values())


def test_server_instructions_route_each_memory_to_its_tool():
    for anchor in (
        "save_personal_memory",
        "save_work_memory",
        "send_message",
        "Write rarely.",
        "at most once",
        "a moment they shared with you, even during work",
    ):
        assert anchor in " ".join(SERVER_INSTRUCTIONS.split())
    assert "save_memory" not in SERVER_INSTRUCTIONS


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
def test_cross_tool_duplicate_is_a_no_op_that_keeps_the_first_kind(rest_in_process):
    content = f"note kinds integration pin {_marker()}: identical content through both tools"
    note_id = build_note_row(content, "work", ["test"], NOW)["id"]
    asyncio.run(_delete(note_id))
    try:
        first = asyncio.run(mcp_server.save_work_memory(content, "natsume", tags=["test"]))
        second = asyncio.run(mcp_server.save_personal_memory(content, "natsume", tags=["test"]))
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
        saved_old = asyncio.run(mcp_server.save_work_memory(old, "natsume", tags=["test"]))
        assert saved_old["id"] == old_id
        saved_new = asyncio.run(
            mcp_server.save_personal_memory(new, "natsume", tags=["test"], supersedes=old_id)
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
