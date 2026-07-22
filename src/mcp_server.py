"""MCP stdio server exposing memory_base's hybrid search as thin tools.

No LLM calls here -- each tool is a direct delegate to search.search()
(FTS + vector + time-decay, fused with RRF, then reranked and
context-restored). See docs/specs §3.2 ⑦.

Register with Claude Code:
    claude mcp add memory-base -- uv --directory <절대경로> run python src/mcp_server.py

Run directly:
    uv run python src/mcp_server.py
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from search import Hit
from search import search as run_search

TEXT_LIMIT = 2000

mcp = FastMCP("memory-base")


def hit_to_dict(hit: Hit) -> dict[str, Any]:
    """Convert a search.Hit into a JSON-serializable dict for tool results."""
    from datetime import datetime, timezone

    out: dict[str, Any] = {
        "source": hit.source,
        "ref": hit.ref,
        "date": datetime.fromtimestamp(hit.ts, tz=timezone.utc).strftime("%Y-%m-%d"),
        "score": hit.rerank_score if hit.rerank_score is not None else hit.rrf,
        "text": hit.text[:TEXT_LIMIT],
    }
    context = hit.meta.get("context")
    if context:
        out["context"] = context
    return out


async def _run_search(query: str, source: str, top_k: int) -> list[dict[str, Any]]:
    hits = await run_search(query, source=source)
    return [hit_to_dict(h) for h in hits[:top_k]]


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
    return await _run_search(query, "all", top_k)


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
    return await _run_search(query, "code", top_k)


@mcp.tool()
async def search_history(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search only past conversation/session history for the given query.

    Use this for retrospective questions like "how did we solve this
    before", "what did we decide about X", or anything referring to prior
    sessions rather than the current codebase. Returns up to `top_k` hits
    sorted by relevance, each with source="history", ref (session ref),
    date (YYYY-MM-DD), score, and text (truncated to 2000 chars).
    """
    return await _run_search(query, "history", top_k)


if __name__ == "__main__":
    mcp.run(transport="stdio")
