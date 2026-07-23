"""MCP server exposing memory_base's hybrid search as thin tools.

No LLM calls here -- each tool is a direct delegate to search.search()
(FTS + vector + time-decay, fused with RRF, then reranked and
context-restored). See docs/specs §3.2 ⑦.

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

import hashlib
import json
import os
import time
from typing import Any, Mapping

import asyncpg
from mcp.server.fastmcp import FastMCP

from memory_base.common import DB_URL, PG_SCHEMA, VllmEmbedder
from memory_base.ingest.history import _embed, _ensure_schema
from memory_base.retrieval.search import Hit
from memory_base.retrieval.search import search as run_search

NOTE_MAX_CHARS = 4000
NOTE_KINDS = ("note", "decision")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765

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


def build_note_row(content: str, kind: str, tags: list[str] | None, now: float) -> dict[str, Any]:
    """Validate a note and map it to memory_chunks columns (no embedding)."""
    content = content.strip()
    if not content:
        raise ValueError("content must not be empty")
    if len(content) > NOTE_MAX_CHARS:
        raise ValueError(f"content exceeds {NOTE_MAX_CHARS} chars")
    if kind not in NOTE_KINDS:
        raise ValueError(f"kind must be one of {NOTE_KINDS}")
    note_id = f"note:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    return {
        "id": note_id,
        "source_type": "agent_note",
        "source_ref": "save_memory",
        "kind": kind,
        "session_id": note_id,
        "raw": content,
        "distilled": content,
        "timestamp": now,
        "idf": None,
        "metadata": {"tags": tags} if tags else {},
    }


@mcp.tool()
async def save_memory(
    content: str, kind: str = "note", tags: list[str] | None = None
) -> dict[str, Any]:
    """Store a distilled memory worth recalling in a future session.

    Save ONLY high-signal content worth remembering later: a decision and its
    rationale, a stated user preference, or a hard-won troubleshooting
    conclusion. Do NOT save running commentary, restatements of the current
    task, or anything trivially re-derivable.

    `content` MUST be written in English regardless of the conversation
    language, and be already distilled (the server does no summarization).
    `kind` is "note" (default) or "decision". `tags` are optional labels.

    Returns {"id", "kind", "stored"}; `stored` is False when identical content
    was already saved (idempotent no-op).
    """
    row = build_note_row(content, kind, tags, time.time())
    embedding = await _embed(VllmEmbedder(), content)
    conn = await asyncpg.connect(DB_URL)
    try:
        await _ensure_schema(conn)
        status = await conn.execute(
            f"""
            INSERT INTO "{PG_SCHEMA}".memory_chunks
              (id, source_type, source_ref, chunk_kind, session_id, content_raw,
               distilled, embedding, ts_last_active, idf_score, metadata)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::halfvec,$9,$10,$11::jsonb)
            ON CONFLICT (id) DO NOTHING
            """,
            row["id"],
            row["source_type"],
            row["source_ref"],
            row["kind"],
            row["session_id"],
            row["raw"],
            row["distilled"],
            embedding,
            row["timestamp"],
            row["idf"],
            json.dumps(row["metadata"], ensure_ascii=False),
        )
    finally:
        await conn.close()
    return {"id": row["id"], "kind": row["kind"], "stored": status.endswith(" 1")}


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
