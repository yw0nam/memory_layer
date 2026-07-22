"""Unit tests for src/mcp_server.py's resolve_transport(): no DB/network access.

resolve_transport() is a pure function of an env mapping, so it's exercised
directly with hand-built dicts (os.environ is never touched).
"""

from __future__ import annotations

import pytest

from memory_base.serve.mcp_server import resolve_transport


def test_resolve_transport_defaults_to_stdio():
    assert resolve_transport({}) == ("stdio", "0.0.0.0", 8765)


def test_resolve_transport_sse():
    assert resolve_transport({"MCP_TRANSPORT": "sse"}) == ("sse", "0.0.0.0", 8765)


def test_resolve_transport_streamable_http():
    result = resolve_transport({"MCP_TRANSPORT": "streamable-http"})
    assert result == ("streamable-http", "0.0.0.0", 8765)


def test_resolve_transport_is_case_insensitive():
    assert resolve_transport({"MCP_TRANSPORT": "SSE"}) == ("sse", "0.0.0.0", 8765)
    assert resolve_transport({"MCP_TRANSPORT": "Stdio"}) == ("stdio", "0.0.0.0", 8765)


def test_resolve_transport_custom_port_and_host():
    result = resolve_transport(
        {"MCP_TRANSPORT": "sse", "MCP_PORT": "9000", "MCP_HOST": "127.0.0.1"}
    )
    assert result == ("sse", "127.0.0.1", 9000)
    assert isinstance(result[2], int)


def test_resolve_transport_invalid_transport_raises_value_error():
    with pytest.raises(ValueError):
        resolve_transport({"MCP_TRANSPORT": "tcp"})


def test_resolve_transport_non_numeric_port_raises_value_error():
    with pytest.raises(ValueError):
        resolve_transport({"MCP_PORT": "not-a-number"})
