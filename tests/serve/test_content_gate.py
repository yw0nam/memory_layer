"""Contract tests for the note content gate (red-first).

Before embedding, every note is judged by the chat model with the prompt of its
kind: a refused note raises ``LowSignalNoteError``, whose message states the
judge's reason and the kind's recovery — move the note to the other save tool
only when it clearly belongs there, one rewrite in total — and offers no
override. A judge failure saves the note stamped ``content_gate: "unavailable"``
(fail-open).

Pure/unit sections follow tests/serve/test_rest_notes.py's FakeConnection
pattern: no DB/network involved.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from starlette.testclient import TestClient

from memory_base.serve import api, mcp_server, notes
from memory_base.serve.mcp_server import SERVER_INSTRUCTIONS
from memory_base.serve.notes import (
    NOTE_GATE_TIMEOUT_SECONDS,
    ContentVerdict,
    JUDGE_PROMPTS,
    LowSignalNoteError,
    judge_note_content,
    save_note,
)

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


def _assert_refusal_text(message: str, reason: str, kind: str) -> None:
    """The refusal states the reason, the other tool, then the rewrite limit; no override."""
    other = "save_work_memory" if kind == "personal" else "save_personal_memory"
    split = "If the reason says to split it"
    assert f"{kind}-memory gate" in message
    assert reason in message
    assert other in message
    assert "send_message" in message
    assert "One rewrite in total, whichever tool" in message
    assert "store nothing" in message
    assert "tell the user" in message
    assert split in message
    after_split = message[message.index(split) :]
    assert "save_personal_memory" in after_split
    assert "save_work_memory" in after_split
    assert message.index(reason) < message.index(other) < message.index("One rewrite in total")
    assert "allow_restatement" not in message


# ---- judge_note_content -----------------------------------------------------


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_judge_builds_the_prompt_of_the_kind_and_parses_the_verdict(monkeypatch, kind):
    captured = {}

    async def fake_chat_json(messages, schema, *, timeout):
        captured["messages"] = messages
        captured["schema"] = schema
        captured["timeout"] = timeout
        return {"accepted": False, "reason": "a progress update on the migration"}

    monkeypatch.setattr(notes, "chat_json", fake_chat_json)
    verdict = asyncio.run(judge_note_content("migrated the parser today", kind))
    assert verdict == ContentVerdict(accepted=False, reason="a progress update on the migration")
    system, user = captured["messages"][0], captured["messages"][-1]
    assert system["role"] == "system"
    assert system["content"] == JUDGE_PROMPTS[kind]
    assert user["role"] == "user"
    assert user["content"] == "migrated the parser today"
    assert captured["schema"] == {
        "type": "object",
        "properties": {
            "accepted": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["accepted", "reason"],
    }
    assert captured["timeout"] == NOTE_GATE_TIMEOUT_SECONDS


# ---- save_note: the gate -----------------------------------------------------


class FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(self, registered: bool = True):
        self._registered = registered
        self.embeds: list[str] = []
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


def test_save_note_refused_note_neither_embeds_nor_inserts(monkeypatch):
    conn = FakeConnection()

    async def refusing_judge(content, kind):
        return ContentVerdict(accepted=False, reason="a progress update")

    _patch_note_deps(monkeypatch, conn, judge=refusing_judge)
    with pytest.raises(LowSignalNoteError) as exc_info:
        asyncio.run(save_note("migrated the parser today", tags=["test"], kind="work"))
    assert exc_info.value.reason == "a progress update"
    _assert_refusal_text(str(exc_info.value), "a progress update", "work")
    assert conn.embeds == []
    assert conn.insert_args is None


def test_a_personal_note_restating_an_artefact_is_refused(monkeypatch):
    conn = FakeConnection()

    async def refusing_judge(content, kind):
        return ContentVerdict(accepted=False, reason="the tracker already records this")

    _patch_note_deps(monkeypatch, conn, judge=refusing_judge)
    with pytest.raises(LowSignalNoteError):
        asyncio.run(save_note("PR #1010 merged today", tags=["test"], kind="personal"))
    assert conn.embeds == []
    assert conn.insert_args is None


def test_judge_failure_stores_the_note_stamped_unavailable(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)

    async def failing_judge(content, kind):
        raise TimeoutError("chat timed out")

    monkeypatch.setattr(notes, "judge_note_content", failing_judge)
    result = asyncio.run(save_note("the parser bug was a stale cache", tags=["test"], kind="work"))
    assert result["stored"] is True
    assert conn.embeds
    metadata = json.loads(conn.insert_args[10])
    assert metadata["content_gate"] == "unavailable"


def test_accepted_note_stamps_no_content_gate_key(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    asyncio.run(save_note("the burst gate weighs recency twice", tags=["test"], kind="work"))
    metadata = json.loads(conn.insert_args[10])
    assert "content_gate" not in metadata


# ---- REST --------------------------------------------------------------------


def test_save_memory_low_signal_note_error_maps_to_409(monkeypatch):
    async def fake_save_note(content, **kwargs):
        raise LowSignalNoteError("a progress update on the migration", "work")

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={"author": "natsume", "content": "migrated the parser today", "kind": "work"},
    )
    assert response.status_code == 409
    body = response.json()
    assert body["reason"] == "a progress update on the migration"
    _assert_refusal_text(body["error"], "a progress update on the migration", "work")


# ---- MCP proxy -----------------------------------------------------------------


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=mcp_server.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(mcp_server, "_client", fake_client)


def test_mcp_save_work_memory_surfaces_the_content_gate_refusal_as_the_tool_error(monkeypatch):
    refusal = LowSignalNoteError("the note restates PR #12", "work")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": str(refusal), "reason": refusal.reason})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(
            mcp_server.save_work_memory("PR #12 changed the parser", "natsume", tags=["test"])
        )
    assert str(exc_info.value) == str(refusal)
    _assert_refusal_text(str(exc_info.value), "the note restates PR #12", "work")


# ---- instructions ----------------------------------------------------------------


def test_server_instructions_state_the_write_policy_and_the_refusal_recovery():
    assert "Write rarely." in SERVER_INSTRUCTIONS
    assert "at most once" in SERVER_INSTRUCTIONS
    assert "allow_restatement" not in SERVER_INSTRUCTIONS
    assert "\n\n\n" not in SERVER_INSTRUCTIONS


def _tools():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.list_tools()
            return {t.name: t for t in result.tools}

    return asyncio.run(_run())


def test_server_instructions_tell_agents_to_search_before_superseding():
    assert "search_memory the same subject" in SERVER_INSTRUCTIONS
    assert "archive them with archive_notes" in SERVER_INSTRUCTIONS


@pytest.mark.parametrize("tool", ["save_personal_memory", "save_work_memory"])
def test_save_tool_description_tells_agents_to_search_before_superseding(tool):
    description = _tools()[tool].description
    assert "search_memory the same subject" in description
    assert "archive them with archive_notes" in description
