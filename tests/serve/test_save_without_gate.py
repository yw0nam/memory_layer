"""Contract tests for storing a note with no chat-model judgement (red-first).

A note that passes the deterministic checks (validation, credential refusal,
near-duplicate refusal, supersede) is embedded and stored; no chat model judges
its content, so a save works with no chat provider at all.

Pure/unit sections follow tests/serve/test_rest_notes.py's FakeConnection
pattern: no DB/network involved.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from memory_base.core import llm
from memory_base.serve import mcp_server
from memory_base.serve.mcp_server import SERVER_INSTRUCTIONS
from memory_base.serve.notes import store
from memory_base.serve.notes.store import save_note


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


@pytest.fixture
def no_chat_provider(monkeypatch):
    """A vLLM provider resolves to a closed port; resolving it or building a client records."""
    calls = []
    resolve = llm.resolve_llm_provider

    def recording_resolve(env):
        calls.append("resolve_llm_provider")
        return resolve(env)

    def unreachable(provider):
        calls.append(provider.name)
        raise ConnectionError("no chat provider")

    for name in ("ZAI_API_KEY", "OPENAI_API_KEY", "CLAUDE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("VLLM_MODEL", "none")
    monkeypatch.setattr(llm, "resolve_llm_provider", recording_resolve)
    monkeypatch.setattr(llm, "_openai_client", unreachable)
    monkeypatch.setattr(llm, "_anthropic_client", unreachable)
    return calls


@pytest.mark.parametrize("kind", ["personal", "work"])
def test_a_note_is_stored_with_no_chat_model_call(monkeypatch, no_chat_provider, kind):
    conn = FakeConnection()
    _patch_note_deps(monkeypatch, conn)
    result = asyncio.run(save_note("migrated the parser today", tags=["test"], kind=kind))
    assert result["stored"] is True
    assert conn.embeds == ["migrated the parser today"]
    assert conn.insert_args is not None
    assert no_chat_provider == []


def _tools():
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.list_tools()
            return {t.name: t for t in result.tools}

    return asyncio.run(_run())


# ---- instructions ----------------------------------------------------------------


def test_server_instructions_state_the_write_policy_without_a_refusal_loop():
    flat = " ".join(SERVER_INSTRUCTIONS.split())
    assert "Write rarely." in flat
    assert (
        "The server stores a note once it passes the validation, credential, and near-duplicate"
        in flat
    )
    for stale in ("at most once", "rewrite", "content gate", "the gate refuses", "low signal"):
        assert stale not in flat
    assert "allow_restatement" not in SERVER_INSTRUCTIONS
    assert "\n\n\n" not in SERVER_INSTRUCTIONS


def test_save_tool_description_states_no_refusal_loop():
    description = " ".join(_tools()["save_memory"].description.split())
    for stale in ("rewrite it once", "refused as low signal", "store nothing"):
        assert stale not in description


def test_server_instructions_tell_agents_to_search_before_superseding():
    assert "search_memory the same subject" in SERVER_INSTRUCTIONS
    assert "archive them with archive_notes" in SERVER_INSTRUCTIONS


def test_save_tool_description_tells_agents_to_search_before_superseding():
    description = _tools()["save_memory"].description
    assert "search_memory the same subject" in description
