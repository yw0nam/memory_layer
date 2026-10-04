"""MCP tools for search: hybrid search over memory and code through the REST API."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from memory_base.serve.common import rest_client


async def _search(
    query: str,
    source: str,
    top_k: int,
    kind: str | None = None,
    tags: list[str] | None = None,
    include_archived: bool = False,
    repo: list[str] | None = None,
    namespace: str | None = None,
    since: str | None = None,
    until: str | None = None,
    min_score: float | None = None,
    author: str | None = None,
    budget_tokens: int | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {"query": query, "source": source, "top_k": top_k}
    if repo is not None:
        body["repo"] = repo
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


async def search_all(
    query: str,
    top_k: int = 10,
    include_archived: bool = False,
    namespace: str | None = None,
    min_score: float | None = None,
    budget_tokens: int | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Search both code and memory for the given query.

    Use this when you don't know or don't need to restrict whether the
    answer lives in the codebase or in stored memory
    (e.g. broad or ambiguous questions). Returns up to `top_k` hits sorted
    by relevance (rerank score, falling back to RRF fusion score); without
    `budget_tokens` the reranked results are capped at 10 before `top_k`
    applies, so a `top_k` above 10 still returns at most 10 hits. Each hit has
    source ("code" or "memory"), ref (file:line-range or document ref),
    date (YYYY-MM-DD), score, text (code hits truncated to 2000 chars; memory
    hits — notes, document chunks, CSV cards — come back whole, already
    bounded when written), repo for code hits, and optional context
    (neighboring code for code hits).

    `include_archived` widens the search to archived memory; use `search_memory`
    for the `kind`/`tags` filters, which apply to memory only.

    `namespace` narrows the memory search to one namespace the caller's API
    key can access; omitted, it covers every namespace the key can access.
    A namespace the key cannot access is rejected by the server.

    Hits scoring below `min_score` are dropped (default 0.25 on the 0-1 rerank
    relevance scale; pass 0 to disable).

    `budget_tokens` (1-32000) switches to budget packing: hits come back in rerank
    order until their estimated size (characters / 4, context included) would
    exceed the budget, and `top_k` and `min_score` are ignored. Use it for
    questions whose answer spans several notes.
    """
    return await _search(
        query,
        "all",
        top_k,
        include_archived=include_archived,
        namespace=namespace,
        min_score=min_score,
        budget_tokens=budget_tokens,
        ctx=ctx,
    )


async def search_code(
    query: str,
    top_k: int = 10,
    repo: list[str] | None = None,
    namespace: str | None = None,
    min_score: float | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Search only the indexed codebase for the given query.

    Use this for questions about code structure, implementation location,
    function/class definitions, or "where is X implemented" style questions.
    Returns up to `top_k` hits sorted by relevance; the reranked results are
    capped at 10 before `top_k` applies, so a `top_k` above 10 still returns at
    most 10 hits. Each hit has source="code",
    repo, ref (file:line-range), date (file mtime as YYYY-MM-DD), score, text
    (truncated to 2000 chars — code chunks have no hard write-time bound, so a
    chunk can run longer than that), and optional context (neighboring code
    chunks for continuity).

    Every cached repository is searched unless `repo` narrows it to the named
    ones; `list_repos` reports the names that exist, and an unknown name simply
    matches nothing. `namespace` has no effect on code (code is not
    namespaced) but is accepted for a uniform tool shape.

    Hits scoring below `min_score` are dropped (default 0.25 on the 0-1 rerank
    relevance scale; pass 0 to disable).
    """
    return await _search(
        query, "code", top_k, repo=repo, namespace=namespace, min_score=min_score, ctx=ctx
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
    author: str | None = None,
    budget_tokens: int | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Search only stored memory for the given query.

    Use this for questions about past decisions, saved notes, ingested
    documents, or any knowledge stored in the memory base rather than
    the current codebase. Returns up to `top_k` hits sorted by relevance;
    without `budget_tokens` the reranked results are capped at 10 before
    `top_k` applies, so a `top_k` above 10 still returns at most 10 hits.
    Each hit has source="memory", ref (document ref), date (YYYY-MM-DD),
    score, and text: the note or chunk in full, never cut in the response
    (bounded when written instead — notes up to 4000 chars, document
    chunks and CSV cards up to 2000).

    Archived memory is excluded by default. Set `include_archived` only when the
    question is explicitly about superseded or historical content; the rows it
    adds carry "archived": true because they may have been replaced by a newer
    note.

    `namespace` narrows the search to one namespace the caller's API key can
    access; omitted, it covers every namespace the key can access. A
    namespace the key cannot access is rejected by the server.

    `kind` is "personal", "work", or "doc" (document chunks and CSV cards).

    `author` narrows the search to notes saved by one agent's author slug,
    e.g. claude-code.

    `since`/`until` bound the search to memory whose event happened in that
    window (its occurred_at, else when it was saved), for
    time-anchored questions ("what did we decide last week"). Both are ISO 8601
    dates or datetimes; a bare date covers that whole day, and naive values are
    read as UTC.

    Hits scoring below `min_score` are dropped (default 0.25 on the 0-1 rerank
    relevance scale; pass 0 to disable).

    `budget_tokens` (1-32000) switches to budget packing: hits come back in rerank
    order until their estimated size (characters / 4, context included) would
    exceed the budget, and `top_k` and `min_score` are ignored. Use it for
    questions whose answer spans several notes.
    """
    return await _search(
        query,
        "memory",
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
