"""MCP server exposing memory_base's REST API as thin tools.

Transport is stdio by default (local dev); set MCP_TRANSPORT=sse|streamable-http
to serve over HTTP instead (e.g. in Docker). MCP_HOST/MCP_PORT control the
bind address (defaults 0.0.0.0:8765).

Register with Claude Code:
    stdio (local):
        claude mcp add memory-base -- uv --directory <absolute-path> run python -m memory_base.serve.mcp_server
    SSE (Docker):
        claude mcp add --transport sse memory-base http://localhost:8765/sse

Run directly:
    uv run python -m memory_base.serve.mcp_server
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx
from mcp.server.fastmcp import FastMCP

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
REST_URL = os.environ.get("REST_URL", "http://localhost:8010")

mcp = FastMCP("memory-base")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=REST_URL)


async def _search(query: str, source: str, top_k: int) -> list[dict[str, Any]]:
    async with _client() as client:
        response = await client.post(
            "/search", json={"query": query, "source": source, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()


@mcp.tool(name="search")
async def search_all(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search both code and conversation history for the given query.

    Use this when you don't know or don't need to restrict whether the
    answer lives in the codebase or in past conversation/session history
    (e.g. broad or ambiguous questions). Returns up to `top_k` hits sorted
    by relevance (rerank score, falling back to RRF fusion score), each with
    source ("code" or "history"), ref (file:line-range or session ref),
    date (YYYY-MM-DD), score, text (truncated to 2000 chars), and optional
    context (neighboring code for code hits).
    """
    return await _search(query, "all", top_k)


@mcp.tool()
async def search_code(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search only the indexed codebase for the given query.

    Use this for questions about code structure, implementation location,
    function/class definitions, or "where is X implemented" style questions.
    Returns up to `top_k` hits sorted by relevance, each with source="code",
    ref (file:line-range), date (file mtime as YYYY-MM-DD), score, text
    (truncated to 2000 chars), and optional context (neighboring code chunks
    for continuity).
    """
    return await _search(query, "code", top_k)


@mcp.tool()
async def search_history(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search only past conversation/session history for the given query.

    Use this for retrospective questions like "how did we solve this
    before", "what did we decide about X", or anything referring to prior
    sessions rather than the current codebase. Returns up to `top_k` hits
    sorted by relevance, each with source="history", ref (session ref),
    date (YYYY-MM-DD), score, and text (truncated to 2000 chars).
    """
    return await _search(query, "history", top_k)


@mcp.tool()
async def save_memory(
    content: str,
    kind: str = "note",
    tags: list[str] | None = None,
    supersedes: str | None = None,
) -> dict[str, Any]:
    """Store a distilled memory worth recalling in a future session.

    Save ONLY high-signal content worth remembering later: a decision and its
    rationale, a stated user preference, or a hard-won troubleshooting
    conclusion. Do NOT save running commentary, restatements of the current
    task, or anything trivially re-derivable.

    `content` MUST be written in English regardless of the conversation
    language, and be already distilled (the server does no summarization).
    `kind` is "note" (default) or "decision". `tags` are optional labels.
    `supersedes` archives an older note; `similar` hints identify related notes.

    `stored` is False when identical content was already saved (idempotent no-op).
    """
    async with _client() as client:
        response = await client.post(
            "/save_memory",
            json={
                "content": content,
                "kind": kind,
                "tags": tags,
                "supersedes": supersedes,
            },
        )
        if response.status_code == 400:
            raise ValueError(response.json()["error"])
        response.raise_for_status()
        return response.json()


def resolve_transport(env: Mapping[str, str]) -> tuple[str, str, int]:
    """Return (transport, host, port) from the MCP environment settings.

    MCP_TRANSPORT defaults to stdio; MCP_HOST defaults to "0.0.0.0"; MCP_PORT
    defaults to 8765. An invalid transport value raises ValueError.
    """
    transport = env.get("MCP_TRANSPORT", "stdio").lower()
    if transport not in ("stdio", "sse", "streamable-http"):
        raise ValueError(f"invalid MCP_TRANSPORT: {transport!r}")
    host = env.get("MCP_HOST", DEFAULT_HOST)
    port_raw = env.get("MCP_PORT", str(DEFAULT_PORT))
    try:
        port = int(port_raw)
    except ValueError as e:
        raise ValueError(f"invalid MCP_PORT: {port_raw!r}") from e
    return transport, host, port


if __name__ == "__main__":
    _transport, _host, _port = resolve_transport(os.environ)
    if _transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = _host
        mcp.settings.port = _port
        mcp.run(transport=_transport)
