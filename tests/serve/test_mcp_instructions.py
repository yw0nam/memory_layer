"""Unit tests for the usage instructions the MCP server advertises: no DB/network access.

The string is only useful if it reaches the client, so the delivery path
(`create_initialization_options()`, which fills the `initialize` response) is
asserted rather than the module constant on its own.
"""

from __future__ import annotations

from memory_base.serve.mcp_server import SERVER_INSTRUCTIONS, mcp


def test_initialize_response_carries_the_instructions():
    options = mcp._mcp_server.create_initialization_options()

    assert options.instructions == SERVER_INSTRUCTIONS
    assert SERVER_INSTRUCTIONS.strip()


def test_instructions_name_the_tools_each_rule_routes_to():
    assert "search_memory" in SERVER_INSTRUCTIONS
    assert "search_code" in SERVER_INSTRUCTIONS
    assert "query_table" in SERVER_INSTRUCTIONS
