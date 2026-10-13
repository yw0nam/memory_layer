"""MCP tools for the turn lane: list sessions, read one, and search the transcript."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from mcp.server.fastmcp import Context

from memory_base.serve.common import rest_client


def _params(**values: Any) -> list[tuple[str, str]]:
    return [(name, str(value)) for name, value in values.items() if value is not None]


async def list_sessions(
    since: str | None = None,
    until: str | None = None,
    origin: str | None = None,
    repo: str | None = None,
    namespace: str | None = None,
    limit: int = 20,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """List conversation sessions the verbatim turn lane holds, newest first.

    Each row is {origin, session_id, namespace, repo, started_at, last_at, turns, title}.
    The lane keeps the user's and the assistant's messages as the client delivered them; it
    is read on demand, never ranked, and never a substitute for notes. Use it to find the
    session of a past decision by time window (`since`, `until`: ISO 8601), `origin`
    (claude_code or hermes), or repo scope (`repo`: repo:<host>/<path>), then read_session.
    `title` is the first user message, cut at 120 characters. `limit` defaults to 20, max 200.
    """
    return await rest_client.call(
        "GET",
        "/sessions",
        params=_params(
            since=since, until=until, origin=origin, repo=repo, namespace=namespace, limit=limit
        ),
        headers=rest_client.auth_headers(ctx),
    )


async def read_session(
    origin: str,
    session_id: str,
    contains: str | None = None,
    offset: int | None = None,
    limit: int = 50,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Read one session's turns verbatim, in order.

    Returns {origin, session_id, namespace, repo, turns, next_offset}; each turn is
    {seq, role, content, reasoning, created_at}. With `contains` (a literal,
    case-insensitive substring of at least 2 characters) only the matching turns come back,
    each with the turn before and after it, overlapping windows merged, matches marked
    `match: true`. `offset` is a seq to start from; when the response is cut by `limit`
    (default 50, max 200) or by size, `next_offset` is the seq to continue from. Coverage
    differs by origin: Claude Code sessions hold the user's prompt and the final answer of
    each turn, no thinking; Hermes sessions hold the turn as Hermes hands it to its memory
    providers (skill scaffolding stripped, secrets masked) with the reasoning of the turn's
    assistant entries. Read it to recover the context of a past decision, or to check a note
    against what was actually said (a note's source_session names its session).
    """
    return await rest_client.call(
        "GET",
        f"/sessions/{quote(origin, safe='')}/{quote(session_id, safe='')}",
        params=_params(contains=contains, offset=offset, limit=limit),
        headers=rest_client.auth_headers(ctx),
    )


async def search_transcript(
    query: str,
    since: str | None = None,
    until: str | None = None,
    origin: str | None = None,
    repo: str | None = None,
    namespace: str | None = None,
    limit: int = 20,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Find turns in the verbatim turn lane whose content holds a literal keyword.

    The match is a case-insensitive substring of at least 2 characters, newest first; it is
    not semantic, and a Korean word with a particle attached still matches. Each hit is
    {origin, session_id, namespace, seq, role, created_at, window}, with a 300-character
    window around the first match and no reasoning; expand a hit with read_session. Narrow
    by time window (`since`, `until`: ISO 8601), `origin` (claude_code or hermes), or repo
    scope (repo:<host>/<path>). `limit` defaults to 20, max 200.
    """
    return await rest_client.call(
        "GET",
        "/turns/search",
        params=_params(
            q=query,
            since=since,
            until=until,
            origin=origin,
            repo=repo,
            namespace=namespace,
            limit=limit,
        ),
        headers=rest_client.auth_headers(ctx),
    )
