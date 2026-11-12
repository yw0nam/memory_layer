"""MCP tools for search: hybrid note search through the REST API."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.server.fastmcp import Context
from pydantic import Field

from memory_base.serve.common import rest_client
from memory_base.serve.common.http import AUTHOR_SCOPE_DESCRIPTION, AuthorScope


async def _search(
    query: str,
    top_k: int,
    kind: str | None = None,
    tags: list[str] | None = None,
    include_archived: bool = False,
    namespace: str | None = None,
    since: str | None = None,
    until: str | None = None,
    min_score: float | None = None,
    author: AuthorScope = None,
    budget_tokens: int | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {"query": query, "top_k": top_k}
    if kind is not None:
        body["kind"] = kind
    if tags is not None:
        body["tags"] = tags
    if include_archived:
        body["include_archived"] = True
    if namespace is not None:
        body["namespaces"] = [namespace]
    if since is not None:
        body["since"] = since
    if until is not None:
        body["until"] = until
    if min_score is not None:
        body["min_score"] = min_score
    if author is not None:
        body["author"] = author
    if budget_tokens is not None:
        body["budget_tokens"] = budget_tokens
    return await rest_client.call(
        "POST",
        "/search",
        json=body,
        headers=rest_client.auth_headers(ctx),
    )


async def search_memory(
    query: str,
    top_k: int = 10,
    kind: str | None = None,
    tags: list[str] | None = None,
    include_archived: bool = False,
    namespace: str | None = None,
    since: str | None = None,
    until: str | None = None,
    min_score: float | None = None,
    author: Annotated[AuthorScope, Field(description=AUTHOR_SCOPE_DESCRIPTION)] = None,
    budget_tokens: int | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Search stored notes for the given query.

    Use this for questions about past decisions and saved notes. Returns up to
    `top_k` hits sorted by relevance; without `budget_tokens` the reranked results are
    capped at 10 before `top_k` applies, so a `top_k` above 10 still returns at most 10
    hits. Each hit has ref, date (YYYY-MM-DD), score, and text: the note in full, never
    cut in the response (bounded when written, up to 1750 chars).

    Archived memory is excluded by default. Set `include_archived` only when the
    question is explicitly about superseded or historical content; the rows it
    adds carry "archived": true because they may have been replaced by a newer
    note.

    `namespace` narrows the search to one namespace; omitted, it covers every
    namespace.

    `kind` is "personal" or "work".

    `since`/`until` bound the search to memory whose event happened in that
    window (its occurred_at, else when it was saved), for
    time-anchored questions ("what did we decide last week"). Both are ISO 8601
    dates or datetimes; a bare date covers that whole day, and naive values are
    read as UTC.

    Hits scoring below `min_score` are dropped (default 0.25 on the 0-1 rerank
    relevance scale; pass 0 to disable).

    `budget_tokens` (1-32000) switches to budget packing: hits come back in rerank
    order until their estimated size (characters / 4) would
    exceed the budget, and `top_k` and `min_score` are ignored. Use it for
    questions whose answer spans several notes.
    """
    return await _search(
        query,
        top_k,
        kind,
        tags,
        include_archived,
        namespace=namespace,
        since=since,
        until=until,
        min_score=min_score,
        author=author,
        budget_tokens=budget_tokens,
        ctx=ctx,
    )
