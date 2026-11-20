"""MCP tools for the session record: list sessions, read one, and search the transcript."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from mcp.server.fastmcp import Context

from memory_base.serve.common import rest_client


def _params(**values: Any) -> dict[str, str]:
    return {name: str(value) for name, value in values.items() if value is not None}


async def list_sessions(
    repo: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 20,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """List your own stored Claude Code sessions, newest first.

    The session record is the verbatim transcript your Claude Code sessions uploaded:
    not memory, read on demand and never ranked. Each row is {session_id, repo, host,
    cwd, started_at, last_at, events, title}; `title` is the first user message cut at
    120 characters. Narrow by repo scope (`repo`: repo:<host>/<path>) or time window
    (`since`, `until`: ISO 8601 date or datetime), then read_session. `limit` defaults
    to 20, max 100. No other author reads your sessions.
    """
    return await rest_client.call(
        "GET",
        "/sessions",
        params=_params(repo=repo, since=since, until=until, limit=limit),
        headers=rest_client.auth_headers(ctx),
    )


async def read_session(
    session_id: str,
    from_seq: int | None = None,
    limit: int = 50,
    role: str | None = None,
    contains: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Read one of your own stored Claude Code sessions verbatim, in order.

    The session record is not memory: it is read on demand and never ranked. Returns
    {session_id, events, next_from_seq}; each event is {seq, role, tool_name, event_at,
    content}, with role user, assistant, tool_call or tool_result. `from_seq` is the seq
    to start from (default 1) and `limit` defaults to 50, max 200. A page also stops
    before 200,000 characters of content (it always holds one event); when events remain,
    `next_from_seq` names the seq to continue from, else it is null.
    `role` keeps one role. With `contains` (a literal, case-insensitive substring of at
    least 2 characters) only the matching events come back, each with the event before
    and after it, matches marked `match: true`. An unknown session gets a 404.
    """
    return await rest_client.call(
        "GET",
        f"/sessions/{quote(session_id, safe='')}",
        params=_params(from_seq=from_seq, limit=limit, role=role, contains=contains),
        headers=rest_client.auth_headers(ctx),
    )


async def search_transcript(
    query: str,
    repo: str | None = None,
    role: str | None = None,
    limit: int = 20,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Find a literal keyword in your own stored Claude Code sessions.

    The session record is not memory: it is read on demand and never ranked. The match
    is a case-insensitive substring of at least 2 characters, newest first; it is not
    semantic, and a Korean word with a particle attached still matches. Each hit is
    {session_id, seq, role, event_at, repo, window}, with a 300-character window around
    the first match; expand a hit with read_session. Narrow by repo scope (`repo`:
    repo:<host>/<path>) or `role` (user, assistant, tool_call, tool_result). `limit`
    defaults to 20, max 50.
    """
    return await rest_client.call(
        "GET",
        "/session_events/search",
        params=_params(q=query, repo=repo, role=role, limit=limit),
        headers=rest_client.auth_headers(ctx),
    )
