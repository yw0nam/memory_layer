"""The MCP tool descriptions state the behaviour the REST API implements."""

from __future__ import annotations

import asyncio

from memory_base.serve import mcp_server


def _descriptions() -> dict[str, str]:
    from mcp.shared.memory import create_connected_server_and_client_session

    async def _run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            result = await client.list_tools()
            return {tool.name: tool.description for tool in result.tools}

    return asyncio.run(_run())


def test_search_tools_state_the_rerank_cap_on_top_k():
    descriptions = _descriptions()
    for name in ("search", "search_code", "search_memory"):
        assert "at most 10" in descriptions[name], name


def test_search_memory_lists_the_accepted_kinds():
    description = _descriptions()["search_memory"]
    for kind in ("doc", "personal", "work"):
        assert f'"{kind}"' in description, kind


def test_query_table_names_the_top_level_columns_field():
    description = _descriptions()["query_table"]
    assert "meta.columns" not in description
    assert "`columns`" in description


def test_list_notes_lists_the_author_field():
    assert "tags, author, namespace" in _descriptions()["list_notes"]


def test_remove_document_states_it_deletes_the_table_rows():
    assert "doc_rows" in _descriptions()["remove_document"]
