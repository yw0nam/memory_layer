"""Contract tests for the save_memory content gate (red-first).

Before embedding, every note is judged against the write policy by the chat
model: an artefact restatement, progress update, file description, or session
narration raises ``LowSignalNoteError``, whose message states the judge's
reason and how to recover — rewrite a fact that exists nowhere else as a note
of its own, retry at most once — and offers no override. A judge failure saves
the note stamped ``content_gate: "unavailable"`` (fail-open). An episode is
judged on provenance alone, so a dated personal event passes and a dated
restatement of a tracker artefact does not.

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
    JUDGE_PROMPT,
    LowSignalNoteError,
    judge_note_content,
    save_note,
)

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


def _assert_refusal_text(message: str, reason: str) -> None:
    """The refusal states the reason, then the rewrite, then the retry limit — no override."""
    assert "content gate" in message
    assert reason in message
    assert "rewrite it to state that fact directly" in message
    assert "save that as a note of its own" in message
    assert "store nothing" in message
    assert "Retry at most once" in message
    assert "do not save it" in message
    assert "tell the user" in message
    assert (
        message.index(reason)
        < message.index("rewrite it to state that fact directly")
        < message.index("Retry at most once")
    )
    assert "allow_restatement" not in message


# ---- judge_note_content -----------------------------------------------------


def test_judge_builds_the_prompt_from_the_write_policy_and_parses_the_verdict(monkeypatch):
    captured = {}

    async def fake_chat_json(messages, schema, *, timeout):
        captured["messages"] = messages
        captured["schema"] = schema
        captured["timeout"] = timeout
        return {"accepted": False, "reason": "a progress update on the migration"}

    monkeypatch.setattr(notes, "chat_json", fake_chat_json)
    verdict = asyncio.run(judge_note_content("migrated the parser today", "note"))
    assert verdict == ContentVerdict(accepted=False, reason="a progress update on the migration")
    system, user = captured["messages"][0], captured["messages"][-1]
    assert system["role"] == "system"
    assert system["content"] == JUDGE_PROMPT
    assert user["role"] == "user"
    assert "migrated the parser today" in user["content"]
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
        asyncio.run(save_note("migrated the parser today", tags=["test"]))
    assert exc_info.value.reason == "a progress update"
    _assert_refusal_text(str(exc_info.value), "a progress update")
    assert conn.embeds == []
    assert conn.insert_args is None


def test_episode_kind_is_judged_with_its_kind(monkeypatch):
    conn = FakeConnection()
    seen = {}

    async def recording_judge(content, kind):
        seen["kind"] = kind
        return ContentVerdict(accepted=True, reason="a dated personal event")

    _patch_note_deps(monkeypatch, conn, judge=recording_judge)
    result = asyncio.run(
        save_note("met to walk through the roadmap", tags=["test"], kind="episode")
    )
    assert seen["kind"] == "episode"
    assert result["stored"] is True


def test_episode_restating_an_artefact_is_refused(monkeypatch):
    conn = FakeConnection()

    async def refusing_judge(content, kind):
        return ContentVerdict(accepted=False, reason="the tracker already records this")

    _patch_note_deps(monkeypatch, conn, judge=refusing_judge)
    with pytest.raises(LowSignalNoteError):
        asyncio.run(save_note("PR #1010 merged today", tags=["test"], kind="episode"))
    assert conn.embeds == []
    assert conn.insert_args is None


def test_judge_failure_stores_the_note_stamped_unavailable(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)

    async def failing_judge(content, kind):
        raise TimeoutError("chat timed out")

    monkeypatch.setattr(notes, "judge_note_content", failing_judge)
    result = asyncio.run(save_note("the parser bug was a stale cache", tags=["test"]))
    assert result["stored"] is True
    assert conn.embeds
    metadata = json.loads(conn.insert_args[10])
    assert metadata["content_gate"] == "unavailable"


def test_accepted_note_stamps_no_content_gate_key(monkeypatch):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    asyncio.run(save_note("the burst gate weighs recency twice", tags=["test"]))
    metadata = json.loads(conn.insert_args[10])
    assert "content_gate" not in metadata


# ---- REST --------------------------------------------------------------------


def test_save_memory_low_signal_note_error_maps_to_409(monkeypatch):
    async def fake_save_note(content, **kwargs):
        raise LowSignalNoteError("a progress update on the migration")

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory", json={"author": "natsume", "content": "migrated the parser today"}
    )
    assert response.status_code == 409
    body = response.json()
    assert body["reason"] == "a progress update on the migration"
    _assert_refusal_text(body["error"], "a progress update on the migration")


# ---- MCP proxy -----------------------------------------------------------------


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=mcp_server.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(mcp_server, "_client", fake_client)


def test_mcp_save_memory_surfaces_the_content_gate_refusal_as_the_tool_error(monkeypatch):
    refusal = LowSignalNoteError("the note restates PR #12")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": str(refusal), "reason": refusal.reason})

    _patch_client(monkeypatch, handler)
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(mcp_server.save_memory("PR #12 changed the parser", "natsume", tags=["test"]))
    assert str(exc_info.value) == str(refusal)
    _assert_refusal_text(str(exc_info.value), "the note restates PR #12")


# ---- instructions ----------------------------------------------------------------


def test_server_instructions_state_the_write_policy_and_the_refusal_recovery():
    assert "Write rarely." in SERVER_INSTRUCTIONS
    assert "at most once" in SERVER_INSTRUCTIONS
    assert "allow_restatement" not in SERVER_INSTRUCTIONS
    assert "\n\n\n" not in SERVER_INSTRUCTIONS


def test_server_instructions_tell_agents_to_search_before_superseding():
    assert "search_memory the same subject" in SERVER_INSTRUCTIONS


def test_judge_prompt_states_one_general_criterion_without_domain_anchors():
    assert "provenance" in JUDGE_PROMPT
    for anchor in ("PR", "commit", "namespace", "file does", "session that produced"):
        assert anchor not in JUDGE_PROMPT
