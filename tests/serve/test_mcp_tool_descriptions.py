"""The MCP tool descriptions state the behaviour the REST API implements."""

from __future__ import annotations

import asyncio

import pytest

from memory_base.retrieval.search import RERANK_TOP
from memory_base.serve import mcp_server


@pytest.fixture(scope="module")
def descriptions() -> dict[str, str]:
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.list_tools()
            return {tool.name: " ".join(tool.description.split()) for tool in result.tools}

    return asyncio.run(_run())


@pytest.fixture(scope="module")
def schemas() -> dict[str, dict]:
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.list_tools()
            return {tool.name: tool.inputSchema for tool in result.tools}

    return asyncio.run(_run())


@pytest.mark.parametrize("name", ["search_memory", "list_notes"])
def test_the_author_argument_states_the_default_and_the_explicit_list(schemas, name):
    author = schemas[name]["properties"]["author"]
    assert author["description"] == (
        "Omitted: your own notes and shared notes. Another agent's notes come back only "
        "when you name it: a list of slugs returns exactly those agents' notes, e.g. "
        '["natsume", "claude-code"]; include your own slug to keep yours. '
        '"all": every agent\'s notes.'
    )
    assert {"type": "array", "items": {"type": "string"}} in author["anyOf"]


def test_the_server_registers_the_note_and_message_tools_only(descriptions):
    assert set(descriptions) == {
        "search_memory",
        "list_notes",
        "save_memory",
        "archive_notes",
        "delete_notes",
        "send_message",
        "list_messages",
        "claim_message",
        "cancel_message",
    }


def test_search_memory_states_the_rerank_cap_on_top_k(descriptions):
    assert f"at most {RERANK_TOP}" in descriptions["search_memory"]


def test_search_memory_lists_the_accepted_kinds(descriptions):
    for kind in ("personal", "work"):
        assert f'"{kind}"' in descriptions["search_memory"], kind


def test_save_memory_states_the_shared_flag(descriptions):
    assert "shared" in descriptions["save_memory"]
