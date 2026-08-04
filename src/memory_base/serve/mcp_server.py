"""MCP server exposing memory_base's REST API as thin tools.

Transport is stdio by default (local dev); set MCP_TRANSPORT=sse|streamable-http
to serve over HTTP instead (e.g. in Docker). MCP_HOST/MCP_PORT control the
bind address (defaults 0.0.0.0:8765).

Register with Claude Code:
    stdio (local):
        claude mcp add memory-base -- uv --directory <absolute-path> run python -m memory_base.serve.mcp_server
    streamable HTTP (Docker):
        claude mcp add --transport http memory-base http://localhost:8765/mcp

Run directly:
    uv run python -m memory_base.serve.mcp_server
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx
from mcp.server.fastmcp import Context, FastMCP

from memory_base.adapters.document import MCP_TEXT_EXTENSIONS
from memory_base.adapters.document import extension_for
from memory_base.core.logger import setup_logging
from memory_base.retrieval.decompose import DEEP_TIMEOUT_SECONDS

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
REST_URL = os.environ.get("REST_URL", "http://localhost:8010")
DEFAULT_NAMESPACE = "default"
NAMESPACE_HEADER = "x-memory-namespaces"

mcp = FastMCP("memory-base")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=REST_URL)


def _raise_backend_error(response: httpx.Response) -> None:
    try:
        message = response.json()["error"]
    except (ValueError, KeyError, TypeError):
        message = f"backend returned {response.status_code}"
    raise ValueError(message)


def _allowed_namespaces(ctx: "Context | None") -> list[str]:
    """The connection's ordered allowed namespace set from X-Memory-Namespaces.

    First entry is the home namespace. No header (or no HTTP request, e.g.
    stdio transport or a direct call outside a session) means ['default'].
    """
    request = None
    if ctx is not None:
        try:
            request = ctx.request_context.request
        except ValueError:
            request = None
    if request is None:
        return [DEFAULT_NAMESPACE]
    header = request.headers.get(NAMESPACE_HEADER)
    if not header:
        return [DEFAULT_NAMESPACE]
    parsed = [part.strip() for part in header.split(",") if part.strip()]
    return parsed or [DEFAULT_NAMESPACE]


def _resolve_read_namespaces(namespace: str | None, ctx: "Context | None") -> list[str]:
    """Search-tool resolution: no argument covers the whole allowed set."""
    allowed = _allowed_namespaces(ctx)
    if namespace is None:
        return allowed
    if namespace not in allowed:
        raise ValueError(f"namespace {namespace!r} is outside the allowed set {allowed}")
    return [namespace]


def _resolve_write_namespace(namespace: str | None, ctx: "Context | None") -> str:
    """Write-tool resolution: no argument goes to the home (first) entry."""
    allowed = _allowed_namespaces(ctx)
    if namespace is None:
        return allowed[0]
    if namespace not in allowed:
        raise ValueError(f"namespace {namespace!r} is outside the allowed set {allowed}")
    return namespace


async def _search(
    query: str,
    source: str,
    top_k: int,
    kind: str | None = None,
    tags: list[str] | None = None,
    include_atoms: bool | None = None,
    include_archived: bool = False,
    repo: list[str] | None = None,
    namespaces: list[str] | None = None,
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {"query": query, "source": source, "top_k": top_k}
    if repo is not None:
        body["repo"] = repo
    if kind is not None:
        body["kind"] = kind
    if tags is not None:
        body["tags"] = tags
    if include_atoms is not None:
        body["include_atoms"] = include_atoms
    if include_archived:
        body["include_archived"] = True
    if namespaces is not None:
        body["namespaces"] = namespaces
    async with _client() as client:
        response = await client.post("/search", json=body)
        if response.status_code == 400:
            _raise_backend_error(response)
        response.raise_for_status()
        return response.json()


@mcp.tool(name="search")
async def search_all(
    query: str,
    top_k: int = 10,
    include_atoms: bool | None = None,
    include_archived: bool = False,
    namespace: str | None = None,
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

    `namespace` narrows the memory search to one namespace within this
    connection's allowed set (see `save_memory`); omitted, it covers the
    whole allowed set. A namespace outside the set raises an error.
    """
    namespaces = _resolve_read_namespaces(namespace, ctx)
    return await _search(
        query,
        "all",
        top_k,
        include_atoms=include_atoms,
        include_archived=include_archived,
        namespaces=namespaces,
    )


@mcp.tool()
async def search_code(
    query: str,
    top_k: int = 10,
    repo: list[str] | None = None,
    namespace: str | None = None,
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
    """
    namespaces = _resolve_read_namespaces(namespace, ctx)
    return await _search(query, "code", top_k, repo=repo, namespaces=namespaces)


@mcp.tool()
async def search_memory(
    query: str,
    top_k: int = 10,
    kind: str | None = None,
    tags: list[str] | None = None,
    include_atoms: bool | None = None,
    include_archived: bool = False,
    namespace: str | None = None,
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

    `namespace` narrows the search to one namespace within this connection's
    allowed set; omitted, it covers the whole allowed set. A namespace outside
    the set raises an error.
    """
    namespaces = _resolve_read_namespaces(namespace, ctx)
    return await _search(
        query, "memory", top_k, kind, tags, include_atoms, include_archived, namespaces=namespaces
    )


@mcp.tool()
async def save_memory(
    content: str,
    kind: str = "note",
    tags: list[str] | None = None,
    supersedes: str | None = None,
    namespace: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Store a distilled memory worth recalling in a future session.

    Save ONLY high-signal content worth remembering later: a decision and its
    rationale, a stated user preference, or a hard-won troubleshooting
    conclusion. Do NOT save running commentary, restatements of the current
    task, or anything trivially re-derivable.

    `content` MUST be written in English regardless of the conversation
    language, and be already distilled (the server does no summarization).
    `kind` is "note" (default) or "decision". `tags` are optional labels.
    `supersedes` archives an older note by id.

    `namespace` picks one namespace from this connection's allowed set;
    omitted, the note lands in the home (first) namespace of that set. A
    namespace outside the set raises an error.

    `stored` is False when identical content was already saved (idempotent no-op).
    """
    resolved_namespace = _resolve_write_namespace(namespace, ctx)
    async with _client() as client:
        response = await client.post(
            "/save_memory",
            json={
                "content": content,
                "kind": kind,
                "tags": tags,
                "supersedes": supersedes,
                "namespace": resolved_namespace,
            },
        )
        if response.status_code == 400:
            _raise_backend_error(response)
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def ingest_document(
    content: str,
    filename: str,
    document_id: str | None = None,
    origin: str | None = None,
    mode: str = "upsert",
    namespace: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Queue a text document for conversion, enrichment, and atomic storage.

    The filename must use a supported text extension: .md, .markdown, .txt,
    .rst, .html, or .htm. Binary documents upload through REST directly.

    `namespace` picks one namespace from this connection's allowed set;
    omitted, the document lands in the home (first) namespace of that set. A
    namespace outside the set raises an error.
    """
    try:
        extension = extension_for(filename)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if extension not in MCP_TEXT_EXTENSIONS:
        raise ValueError("MCP document ingestion supports text formats only")
    resolved_namespace = _resolve_write_namespace(namespace, ctx)
    data = {"filename": filename, "mode": mode, "namespace": resolved_namespace}
    if document_id is not None:
        data["document_id"] = document_id
    if origin is not None:
        data["origin"] = origin
    async with _client() as client:
        response = await client.post(
            "/ingest/document",
            data=data,
            files={"file": (filename, content.encode("utf-8"))},
        )
        if response.status_code in {400, 413, 415, 429}:
            _raise_backend_error(response)
        response.raise_for_status()
        payload = response.json()
        return {"job_id": payload["job_id"], "status_url": payload["status_url"]}


@mcp.tool()
async def ingest_repo(
    url: str,
    branch: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Clone (or re-sync) a git repository into the code index.

    `url` is an http(s)/ssh git URL or the git@host:path SSH form. `branch`
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
    async with _client() as client:
        response = await client.post("/repos", json=body)
        if response.status_code in {400, 429}:
            _raise_backend_error(response)
        response.raise_for_status()
        payload = response.json()
        return {"job_id": payload["job_id"], "status_url": payload["status_url"]}


@mcp.tool()
async def remove_repo(name: str) -> dict[str, Any]:
    """Remove a repository from the code index by its cache name.

    Queues a re-index that tears down the removed repo's code chunks. Returns
    {job_id, status_url}; poll status_url for progress.
    """
    async with _client() as client:
        response = await client.delete(f"/repos/{name}")
        if response.status_code in {400, 404, 429}:
            _raise_backend_error(response)
        response.raise_for_status()
        payload = response.json()
        return {"job_id": payload["job_id"], "status_url": payload["status_url"]}


@mcp.tool()
async def list_repos() -> list[dict[str, Any]]:
    """List indexed repositories.

    Returns one entry per cached repo with name, origin url, current branch,
    short head commit, and the number of indexed code chunks.
    """
    async with _client() as client:
        response = await client.get("/repos")
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def deep_search(
    query: str,
    max_hops: int | None = None,
    kind: str | None = None,
    tags: list[str] | None = None,
    include_archived: bool = False,
    namespace: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Multi-hop decomposition over stored memory for complex questions.

    Use when a single search is unlikely to answer a multi-part or
    multi-hop question (e.g. "which service owned the incident that
    caused the checkout latency fix?"). Operates on memory only.
    Returns evidence entries with ref, text, kind, tags, date, hop,
    atom_question, and id; a trace of sub-questions per hop; and
    hops_used and stopped_reason. Archived evidence carries "archived": true.

    `include_archived` carries the same caveats as in `search_memory`.

    `namespace` narrows retrieval to one namespace within this connection's
    allowed set; omitted, it covers the whole allowed set. A namespace
    outside the set raises an error.
    """
    namespaces = _resolve_read_namespaces(namespace, ctx)
    body: dict[str, Any] = {"query": query, "namespaces": namespaces}
    if max_hops is not None:
        body["max_hops"] = max_hops
    if kind is not None:
        body["kind"] = kind
    if tags is not None:
        body["tags"] = tags
    if include_archived:
        body["include_archived"] = True
    # +30s covers HTTP/round-trip slack beyond the server-side deep-search deadline.
    timeout = DEEP_TIMEOUT_SECONDS + 30
    async with _client() as client:
        response = await client.post("/search/deep", json=body, timeout=timeout)
        if response.status_code == 400:
            _raise_backend_error(response)
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
    setup_logging()
    _transport, _host, _port = resolve_transport(os.environ)
    if _transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = _host
        mcp.settings.port = _port
        mcp.run(transport=_transport)
