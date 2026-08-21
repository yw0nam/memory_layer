"""MCP server exposing memory_base's REST API as thin tools.

Transport is stdio by default (local dev); set MCP_TRANSPORT=sse|streamable-http
to serve over HTTP instead (e.g. in Docker). MCP_HOST/MCP_PORT control the
bind address (defaults 0.0.0.0:8765).

Every REST call carries an X-API-Key: over streamable HTTP it is read from
the incoming MCP request's own X-API-Key header and forwarded verbatim; over
stdio (no HTTP request to read from) it comes from the MEMORY_API_KEY
environment variable.

Register with Claude Code:
    stdio (local):
        claude mcp add memory-base --env MEMORY_API_KEY=<key> -- \\
          uv --directory <absolute-path> run python -m memory_base.serve.mcp_server
    streamable HTTP (Docker):
        claude mcp add --transport http memory-base http://localhost:8765/mcp \\
          --header "X-API-Key: <key>"

Run directly:
    uv run python -m memory_base.serve.mcp_server
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from memory_base.adapters.document import MCP_TEXT_EXTENSIONS
from memory_base.adapters.document import extension_for
from memory_base.core.logger import setup_logging

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
REST_URL = os.environ.get("REST_URL", "http://localhost:8010")
API_KEY_HEADER = "x-api-key"

# Served in the initialize response, so it is stated once per client session:
# the store's invariants only. Per-consumer usage belongs to the consumer.
SERVER_INSTRUCTIONS = """\
memory-base holds only distilled knowledge, in three lanes: notes (why something was
decided), code (indexed repositories), and table rows (numbers, read with SQL).

Read first. Before starting a task or answering from recall, search_memory for earlier
decisions on the subject — arriving without them is this server's most common misuse.
search_code spans every indexed repository, not only the one in front of you. Questions
about numbers are computed, not retrieved: search finds the card, query_table computes
over the rows, and search never returns the rows themselves.

Write rarely. A note earns its place when it captures what the next session would
otherwise have to rediscover: a decision and the alternatives it rejected, a reproduced
bug with its known fix, a non-obvious environment fact, an approach that failed and why.
The status of a PR or issue, progress updates, and descriptions of what a file does are
none of those — git and search_code already answer them, and stale copies only dilute
retrieval. When a note goes out of date, supersede it rather than adding a second note
that contradicts it.

Work knowledge belongs in the key's home namespace. Personal context — schedule,
relationships, private preferences — belongs in a private namespace, never the shared
one. A note's first tag names its subject, usually the repository or domain it belongs
to, so that a later search can narrow to it."""


def resolve_transport_security(
    env: Mapping[str, str],
) -> TransportSecuritySettings | None:
    """Return configured transport security or defer to FastMCP defaults."""
    allowed_hosts = [
        host.strip() for host in env.get("MCP_ALLOWED_HOSTS", "").split(",") if host.strip()
    ]
    if not allowed_hosts:
        return None
    return TransportSecuritySettings(allowed_hosts=allowed_hosts)


mcp = FastMCP(
    "memory-base",
    instructions=SERVER_INSTRUCTIONS,
    transport_security=resolve_transport_security(os.environ),
)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=REST_URL)


def _raise_backend_error(response: httpx.Response) -> None:
    try:
        message = response.json()["error"]
    except (ValueError, KeyError, TypeError):
        message = f"backend returned {response.status_code}"
    raise ValueError(message)


async def _call(
    method: str,
    path: str,
    *,
    expect_errors: frozenset[int] = frozenset({401, 403}),
    **kwargs: Any,
) -> Any:
    """Issue a REST call and return the decoded JSON body.

    Statuses in `expect_errors` raise via `_raise_backend_error` (backend
    error payload); any other non-2xx raises through `raise_for_status`.
    """
    async with _client() as client:
        response = await client.request(method, path, **kwargs)
        if response.status_code in expect_errors:
            _raise_backend_error(response)
        response.raise_for_status()
        return response.json()


def _api_key(ctx: "Context | None") -> str | None:
    """The caller's API key: the request's X-API-Key header, or MEMORY_API_KEY for stdio."""
    request = None
    if ctx is not None:
        try:
            request = ctx.request_context.request
        except ValueError:
            request = None
    if request is not None:
        header = request.headers.get(API_KEY_HEADER)
        if header:
            return header
    return os.environ.get("MEMORY_API_KEY")


def _auth_headers(ctx: "Context | None") -> dict[str, str]:
    api_key = _api_key(ctx)
    return {API_KEY_HEADER: api_key} if api_key else {}


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
    ctx: "Context | None" = None,
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
    return await _call(
        "POST",
        "/search",
        json=body,
        headers=_auth_headers(ctx),
        expect_errors=frozenset({400, 401, 403}),
    )


@mcp.tool(name="search")
async def search_all(
    query: str,
    top_k: int = 10,
    include_archived: bool = False,
    namespace: str | None = None,
    min_score: float | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Search both code and memory for the given query.

    Use this when you don't know or don't need to restrict whether the
    answer lives in the codebase or in stored memory
    (e.g. broad or ambiguous questions). Returns up to `top_k` hits sorted
    by relevance (rerank score, falling back to RRF fusion score), each with
    source ("code" or "memory"), ref (file:line-range or document ref),
    date (YYYY-MM-DD), score, text (truncated to 2000 chars), repo for code
    hits, and optional context (neighboring code for code hits).

    `include_archived` widens the search to archived memory; use `search_memory`
    for the `kind`/`tags` filters, which apply to memory only.

    `namespace` narrows the memory search to one namespace the caller's API
    key can access; omitted, it covers every namespace the key can access.
    A namespace the key cannot access is rejected by the server.

    Hits scoring below `min_score` are dropped (default 0.25 on the 0-1 rerank
    relevance scale; pass 0 to disable).
    """
    return await _search(
        query,
        "all",
        top_k,
        include_archived=include_archived,
        namespace=namespace,
        min_score=min_score,
        ctx=ctx,
    )


@mcp.tool()
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
    Returns up to `top_k` hits sorted by relevance, each with source="code",
    repo, ref (file:line-range), date (file mtime as YYYY-MM-DD), score, text
    (truncated to 2000 chars), and optional context (neighboring code chunks
    for continuity).

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


@mcp.tool()
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
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """Search only stored memory for the given query.

    Use this for questions about past decisions, saved notes, ingested
    documents, or any knowledge stored in the memory base rather than
    the current codebase. Returns up to `top_k` hits sorted by relevance,
    each with source="memory", ref (document ref), date (YYYY-MM-DD),
    score, and text (truncated to 2000 chars).

    Archived memory is excluded by default. Set `include_archived` only when the
    question is explicitly about superseded or historical content: it also drops
    recency weighting, and the rows it adds carry "archived": true because they
    may have been replaced by a newer note.

    `namespace` narrows the search to one namespace the caller's API key can
    access; omitted, it covers every namespace the key can access. A
    namespace the key cannot access is rejected by the server.

    `since`/`until` bound the search to memory last active in that window, for
    time-anchored questions ("what did we decide last week"). Both are ISO 8601
    dates or datetimes; a bare date covers that whole day, and naive values are
    read as UTC.

    Hits scoring below `min_score` are dropped (default 0.25 on the 0-1 rerank
    relevance scale; pass 0 to disable).
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
        ctx=ctx,
    )


@mcp.tool()
async def list_notes(
    tags: list[str] | None = None,
    kind: str | None = None,
    since: str | None = None,
    until: str | None = None,
    include_archived: bool = False,
    namespace: str | None = None,
    limit: int | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """List stored notes deterministically — no search query, no relevance ranking.

    Use this for exact reads a similarity search cannot promise to be complete:
    every note carrying a tag (e.g. loading profile or preference notes at
    session start), or every note saved in a time window ("what was saved last
    week"). Works without the embedding backend. Returns up to `limit` notes
    (default 50, max 200) newest-first, each with id, kind, text (truncated to
    2000 chars), tags, namespace, and date (YYYY-MM-DD).

    `tags` matches notes carrying any of the given tags. `kind` is "note" or
    "decision". `since`/`until` are ISO 8601 dates or datetimes (a bare date
    covers that whole day; naive values are read as UTC). `include_archived`
    adds superseded notes, marked "archived": true. `namespace` narrows to one
    namespace the caller's API key can access; omitted, it covers every
    namespace the key can access.
    """
    params: list[tuple[str, str]] = [("tags", tag) for tag in tags or []]
    if kind is not None:
        params.append(("kind", kind))
    if since is not None:
        params.append(("since", since))
    if until is not None:
        params.append(("until", until))
    if include_archived:
        params.append(("include_archived", "true"))
    if namespace is not None:
        params.append(("namespace", namespace))
    if limit is not None:
        params.append(("limit", str(limit)))
    return await _call(
        "GET",
        "/notes",
        params=params,
        headers=_auth_headers(ctx),
        expect_errors=frozenset({400, 401, 403}),
    )


@mcp.tool()
async def save_memory(
    content: str,
    kind: str = "note",
    tags: list[str] | None = None,
    supersedes: str | None = None,
    namespace: str | None = None,
    occurred_at: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Store a distilled memory worth recalling in a future session.

    Save ONLY high-signal content worth remembering later: a decision and its
    rationale, a stated user preference, or a hard-won troubleshooting
    conclusion. Do NOT save running commentary, restatements of the current
    task, or anything trivially re-derivable.

    `content` MUST be written in English regardless of the conversation
    language, and be already distilled (the server does no summarization).
    `kind` is "note" (default), "decision", or "episode" (a 1-3 sentence
    past-tense record of one conversation session). `tags` are optional
    labels. `supersedes` archives an older note by id.

    `occurred_at` backdates the stored timestamp to an ISO 8601 date or
    datetime instead of now, e.g. to land a backfilled episode on the day it
    happened; a future or unparseable value is rejected.

    `namespace` picks one namespace the caller's API key can access; omitted,
    the note lands in the key's home namespace. A namespace the key cannot
    access is rejected by the server.

    `stored` is False when identical content was already saved (idempotent no-op).
    """
    body: dict[str, Any] = {
        "content": content,
        "kind": kind,
        "tags": tags,
        "supersedes": supersedes,
    }
    if namespace is not None:
        body["namespace"] = namespace
    if occurred_at is not None:
        body["occurred_at"] = occurred_at
    return await _call(
        "POST",
        "/save_memory",
        json=body,
        headers=_auth_headers(ctx),
        expect_errors=frozenset({400, 401, 403}),
    )


@mcp.tool()
async def query_table(
    sql: str,
    namespace: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Run a read-only SQL query over ingested CSV rows.

    Find the CSV card with `search_memory` first. Its `source_ref` is the
    `document_id`, and its meta.columns lists the available JSON keys. Rows
    live in `memory.doc_rows` as jsonb: use `(data->>'column')::numeric` for
    numeric calculations and `WHERE document_id = '...'` to scope one table.
    The server restricts the query to one permitted namespace and returns at
    most 1,000 rows.
    """
    body: dict[str, Any] = {"sql": sql}
    if namespace is not None:
        body["namespace"] = namespace
    return await _call(
        "POST",
        "/tables/query",
        json=body,
        headers=_auth_headers(ctx),
        expect_errors=frozenset({400, 401, 403, 408, 413}),
    )


@mcp.tool()
async def ingest_document(
    content: str,
    filename: str,
    document_id: str | None = None,
    origin: str | None = None,
    mode: str = "upsert",
    namespace: str | None = None,
    tags: list[str] | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Queue a text document for conversion, chunking, and atomic storage.

    The filename must use a supported text extension: .md, .markdown, .txt,
    .rst, .html, .htm, or .csv. Binary documents upload through REST directly.

    `namespace` picks one namespace the caller's API key can access; omitted,
    the document lands in the key's home namespace. A namespace the key
    cannot access is rejected by the server.

    `tags` are lowercased topical labels stamped on every chunk of the
    document and usable as the `tags` search filter.
    """
    try:
        extension = extension_for(filename)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if extension not in MCP_TEXT_EXTENSIONS:
        raise ValueError("MCP document ingestion supports text formats only")
    data: dict[str, Any] = {"filename": filename, "mode": mode}
    if namespace is not None:
        data["namespace"] = namespace
    if tags:
        data["tags"] = tags
    if document_id is not None:
        data["document_id"] = document_id
    if origin is not None:
        data["origin"] = origin
    payload = await _call(
        "POST",
        "/ingest/document",
        data=data,
        files={"file": (filename, content.encode("utf-8"))},
        headers=_auth_headers(ctx),
        expect_errors=frozenset({400, 401, 403, 413, 415, 429}),
    )
    return {"job_id": payload["job_id"], "status_url": payload["status_url"]}


@mcp.tool()
async def remove_document(
    document_id: str, namespace: str | None = None, ctx: Context | None = None
) -> dict[str, Any]:
    """Delete a document's stored chunks from one namespace by its document_id.

    Restricted to an admin key or the document's creator (the key that first
    ingested it); a non-creator, non-admin caller gets a 403, and an unknown
    document in that namespace gets a 404. `namespace` defaults to the
    caller's home namespace. Returns {document_id, namespace, deleted} where
    `deleted` is the number of chunk rows removed.
    """
    params = {"namespace": namespace} if namespace is not None else None
    return await _call(
        "DELETE",
        f"/ingest/documents/{document_id}",
        params=params,
        headers=_auth_headers(ctx),
        expect_errors=frozenset({400, 401, 403, 404}),
    )


@mcp.tool()
async def ingest_repo(
    url: str,
    branch: str | None = None,
    name: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Clone (or re-sync) a git repository into the code index.

    `url` is an http(s) git URL with no embedded credentials. `branch`
    selects the branch on the initial clone only. `name` overrides the cache
    directory name (derived from the URL basename by default). Re-issuing this
    for an existing name fast-forwards its current branch instead of
    re-cloning, ignoring `branch` — remove and re-add the repo to switch
    branch. Returns {job_id, status_url}; poll status_url for progress.
    """
    body: dict[str, Any] = {"url": url}
    if branch is not None:
        body["branch"] = branch
    if name is not None:
        body["name"] = name
    payload = await _call(
        "POST",
        "/repos",
        json=body,
        headers=_auth_headers(ctx),
        expect_errors=frozenset({400, 401, 403, 429}),
    )
    return {"job_id": payload["job_id"], "status_url": payload["status_url"]}


@mcp.tool()
async def remove_repo(name: str, ctx: Context | None = None) -> dict[str, Any]:
    """Remove a repository from the code index by its cache name.

    Restricted to an admin key or the repo's owner (the key that first
    ingested it); a non-owner, non-admin caller gets a 403. Queues a re-index
    that tears down the removed repo's code chunks. Returns
    {job_id, status_url}; poll status_url for progress.
    """
    payload = await _call(
        "DELETE",
        f"/repos/{name}",
        headers=_auth_headers(ctx),
        expect_errors=frozenset({400, 401, 403, 404, 429}),
    )
    return {"job_id": payload["job_id"], "status_url": payload["status_url"]}


@mcp.tool()
async def list_repos(ctx: Context | None = None) -> list[dict[str, Any]]:
    """List indexed repositories.

    Returns one entry per cached repo with name, origin url, current branch,
    short head commit, the number of indexed code chunks, and the owning
    key's label (null when unrecorded).
    """
    return await _call("GET", "/repos", headers=_auth_headers(ctx))


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
    setup_logging()
    _transport, _host, _port = resolve_transport(os.environ)
    if _transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = _host
        mcp.settings.port = _port
        mcp.run(transport=_transport)
