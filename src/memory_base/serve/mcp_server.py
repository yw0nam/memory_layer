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

import logging
import os
from typing import Any, Mapping

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from memory_base.adapters.document import MCP_TEXT_EXTENSIONS
from memory_base.adapters.document import extension_for
from memory_base.core.logger import setup_logging
from memory_base.serve.common import rest_client
from memory_base.serve.messages import tools as message_tools
from memory_base.serve.notes import tools as note_tools
from memory_base.serve.profiles import tools as profile_tools

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765

# Served in the initialize response, so it is stated once per client session:
# the store's invariants only. Per-consumer usage belongs to the consumer.
_SERVER_INSTRUCTIONS_OPENING = """\
memory-base holds distilled knowledge in three lanes — notes (memory of the user and of
the work), code (indexed repositories), and table rows (numbers, read with SQL) — plus
an addressed message lane for one-time signals between sessions (never embedded, never
searched).

Read first. Before starting a task or answering from recall, search_memory for earlier
decisions on the subject — arriving without them is this server's most common misuse.
search_code spans every indexed repository, not only the one in front of you. Questions
about numbers are computed, not retrieved: search finds the card, query_table computes
over the rows, and search never returns the rows themselves.

"""

_WRITE_POLICY = """\
Write rarely. Save with save_memory and label the note with kind: "personal" for something
about the user (their life, their day, or a moment they shared with you, even during
work), "work" for work knowledge that code, version control, and the tracker cannot
answer; progress or state for the next session goes to send_message. The kind only labels
a note and never decides whether it is stored. The server stores a note once it passes
the validation, credential, and near-duplicate checks, so what is worth keeping is your
call. Before saving, search_memory the same subject; when the new note replaces one,
supersede it rather than adding a note that contradicts it, and if other active notes
state the same stale value, archive them with archive_notes."""

_MESSAGE_LANE = """\
Messages are an addressed, one-time signal lane beside the notes: never embedded, never
searchable, listed while pending and consumed by claiming. At the start of a session,
list_messages for your namespaces and claim_message each one you act on — a claim is
exclusive, and it fires only when called, never automatically from a prefetch hook.
send_message takes two shapes: a general message (status "info", no scope) addressed to
a namespace, or — with a scope repo:<origin> or project:<organization>/<project> — a
handoff, the latest snapshot of a work state statused in_progress, blocked, or
completed. Whoever next works in that scope claims it; a new snapshot supersedes the
pending one, and a completed handoff remains the delivered record of that state. A
handoff stays pending until it is claimed, superseded, or cancelled unless its sender
gives an expires_at; a general message expires after the server's default TTL. Keep the
lanes straight: a note is durable knowledge, memory of the user or of the work, read
again whenever it matches; a message is operational state and is consumed once. That
makes a message the right carrier for progress and next steps, and the wrong place for
anything meant to be read more than once."""

_SERVER_INSTRUCTIONS_CLOSING = """\
Work knowledge belongs in the key's home namespace. Personal context — schedule,
relationships, private preferences — belongs in a private namespace, never the shared
one. A note's first tag names its subject, usually the repository or domain it belongs
to, so that a later search can narrow to it.

Curate rarely. list_memory_duplicates shows active note pairs whose meaning nearly
coincides; read both sides, then either merge them into one note with save_memory
(supersedes=...) or drop one with archive_notes. Every write and archive names its author. delete_notes is for rows that
must never resurface; archiving is otherwise always preferred."""

SERVER_INSTRUCTIONS = "\n\n".join(
    (
        _SERVER_INSTRUCTIONS_OPENING.rstrip("\n"),
        _WRITE_POLICY,
        _MESSAGE_LANE,
        _SERVER_INSTRUCTIONS_CLOSING,
    )
)


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


@mcp.tool(name="search")
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


mcp.tool()(note_tools.list_notes)
mcp.tool()(note_tools.save_memory)
mcp.tool()(message_tools.send_message)
mcp.tool()(message_tools.list_messages)
mcp.tool()(message_tools.claim_message)
mcp.tool()(message_tools.cancel_message)
mcp.tool()(profile_tools.update_my_profile)
mcp.tool()(profile_tools.propose_user_profile)
mcp.tool()(note_tools.list_memory_duplicates)
mcp.tool()(note_tools.archive_notes)
mcp.tool()(note_tools.restore_notes)
mcp.tool()(note_tools.delete_notes)


@mcp.tool()
async def query_table(
    sql: str,
    namespace: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Run a read-only SQL query over ingested CSV rows.

    Find the CSV card with `search_memory` first. Its `ref` is
    `<document_id>#card-N` (the part before `#` is the document_id), and its
    top-level `columns` field lists the available JSON keys. Rows live in `memory.doc_rows`
    as jsonb: use `(data->>'column')::numeric` for numeric calculations and
    `WHERE document_id = '...'` to scope one table.
    The server restricts the query to one permitted namespace and returns at
    most 1,000 rows.
    """
    body: dict[str, Any] = {"sql": sql}
    if namespace is not None:
        body["namespace"] = namespace
    return await rest_client.call(
        "POST",
        "/tables/query",
        json=body,
        headers=rest_client.auth_headers(ctx),
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

    A document carrying a credential (an API key, token, private key, JWT, or
    password in a URL) is refused whole: in its filename, `document_id`,
    `origin`, or tags the call fails; in its content the job fails and
    nothing from it is stored.
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
    payload = await rest_client.call(
        "POST",
        "/ingest/document",
        data=data,
        files={"file": (filename, content.encode("utf-8"))},
        headers=rest_client.auth_headers(ctx),
    )
    return {"job_id": payload["job_id"], "status_url": payload["status_url"]}


@mcp.tool()
async def remove_document(
    document_id: str, namespace: str | None = None, ctx: Context | None = None
) -> dict[str, Any]:
    """Delete a document's stored chunks and table rows from one namespace by its document_id.

    Restricted to an admin key or the document's creator (the key that first
    ingested it); a non-creator, non-admin caller gets a 403, and an unknown
    document in that namespace gets a 404. `namespace` defaults to the
    caller's home namespace. Returns {document_id, namespace, deleted} where
    `deleted` is the number of chunk rows removed; a tabular document's
    `doc_rows` rows are removed with them.
    """
    params = {"namespace": namespace} if namespace is not None else None
    return await rest_client.call(
        "DELETE",
        f"/ingest/documents/{document_id}",
        params=params,
        headers=rest_client.auth_headers(ctx),
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
    payload = await rest_client.call(
        "POST",
        "/repos",
        json=body,
        headers=rest_client.auth_headers(ctx),
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
    payload = await rest_client.call(
        "DELETE",
        f"/repos/{name}",
        headers=rest_client.auth_headers(ctx),
    )
    return {"job_id": payload["job_id"], "status_url": payload["status_url"]}


@mcp.tool()
async def list_repos(ctx: Context | None = None) -> list[dict[str, Any]]:
    """List indexed repositories.

    Returns one entry per cached repo with name, origin url, current branch,
    short head commit, the number of indexed code chunks, and the owning
    key's label (null when unrecorded).
    """
    return await rest_client.call("GET", "/repos", headers=rest_client.auth_headers(ctx))


def quiet_request_noise() -> None:
    """Routine Ping/ListTools request lines drown real events at INFO."""
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)


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
    quiet_request_noise()
    _transport, _host, _port = resolve_transport(os.environ)
    if _transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = _host
        mcp.settings.port = _port
        mcp.run(transport=_transport)
