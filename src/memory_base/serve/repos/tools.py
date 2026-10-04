"""MCP tools for repositories: queue an ingest or removal and list the indexed repos."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from memory_base.serve.common import rest_client


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


async def list_repos(ctx: Context | None = None) -> list[dict[str, Any]]:
    """List indexed repositories.

    Returns one entry per cached repo with name, origin url, current branch,
    short head commit, the number of indexed code chunks, and the owning
    key's label (null when unrecorded).
    """
    return await rest_client.call("GET", "/repos", headers=rest_client.auth_headers(ctx))
