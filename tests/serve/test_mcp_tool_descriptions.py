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


def test_search_tools_state_the_rerank_cap_on_top_k(descriptions):
    for name in ("search", "search_code", "search_memory"):
        assert f"at most {RERANK_TOP}" in descriptions[name], name


def test_search_memory_lists_the_accepted_kinds(descriptions):
    for kind in ("doc", "personal", "work"):
        assert f'"{kind}"' in descriptions["search_memory"], kind


def test_query_table_names_the_top_level_columns_field(descriptions):
    assert "meta.columns" not in descriptions["query_table"]
    assert "`columns`" in descriptions["query_table"]


def test_list_notes_mentions_author(descriptions):
    assert "author" in descriptions["list_notes"]


def test_remove_document_states_it_deletes_the_table_rows(descriptions):
    assert "doc_rows" in descriptions["remove_document"]
