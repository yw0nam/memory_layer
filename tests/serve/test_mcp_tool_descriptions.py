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


@pytest.mark.parametrize(
    "name", ["search_memory", "list_notes", "list_memory_duplicates", "read_entity"]
)
def test_the_author_argument_states_the_default_and_the_explicit_list(schemas, name):
    author = schemas[name]["properties"]["author"]
    assert author["description"] == (
        "Omitted: your own notes and shared notes. Another agent's notes come back only "
        'when you name it: a list of slugs returns exactly those agents\' notes, e.g. '
        '["natsume", "claude-code"]; include your own slug to keep yours. '
        '"all": every agent\'s notes.'
    )
    assert {"type": "array", "items": {"type": "string"}} in author["anyOf"]


def test_search_tools_state_the_rerank_cap_on_top_k(descriptions):
    for name in ("search", "search_code", "search_memory"):
        assert f"at most {RERANK_TOP}" in descriptions[name], name


def test_search_memory_lists_the_accepted_kinds(descriptions):
    for kind in ("doc", "personal", "work"):
        assert f'"{kind}"' in descriptions["search_memory"], kind


def test_query_table_names_the_top_level_columns_field(descriptions):
    assert "meta.columns" not in descriptions["query_table"]
    assert "`columns`" in descriptions["query_table"]


def test_save_memory_states_the_shared_flag(descriptions):
    assert "shared" in descriptions["save_memory"]


def test_search_states_note_hits_are_own_and_shared(descriptions):
    assert "your own and shared notes" in descriptions["search"]


def test_remove_document_states_it_deletes_the_table_rows(descriptions):
    assert "doc_rows" in descriptions["remove_document"]


def test_turn_tools_state_the_lane_and_each_origins_coverage(descriptions):
    assert "never a substitute for notes" in descriptions["list_sessions"]
    for phrase in ("no thinking", "reasoning", "next_offset", "match"):
        assert phrase in descriptions["read_session"], phrase
    for name in ("list_sessions", "read_session", "search_transcript"):
        assert len(descriptions[name]) < 1500, name
    assert "particle" in descriptions["search_transcript"]
    assert "source_session" in descriptions["read_session"]


def test_entity_tools_state_the_card_the_one_writer_and_the_correction_path(descriptions):
    card = descriptions["read_entity"]
    assert len(card) < 1500
    for phrase in ("consolidation", "saving a note", "never a direction"):
        assert phrase in card, phrase
    assert "Don't remember" in descriptions["register_entity"]
    assert "consolidation run" in mcp_server.SERVER_INSTRUCTIONS
