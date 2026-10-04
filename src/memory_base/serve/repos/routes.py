"""REST routes for repositories: queue an ingest or removal, list the cache, and report job state."""

from __future__ import annotations

import shutil
import uuid

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.serve.common import job_store
from memory_base.serve.common.http import error, json_body
from memory_base.serve.repos import cache


def _low_on_disk() -> bool:
    """True when the cache volume cannot hold a full-size checkout above the headroom floor.

    Walks up to the nearest existing parent when the cache dir is not yet
    created; an unreadable volume counts as low.
    """
    path = cache.CACHE_ROOT
    while not path.exists():
        parent = path.parent
        if parent == path:
            return False
        path = parent
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return True
    return usage.free < cache.DISK_HEADROOM_BYTES + cache.REPO_MAX_BYTES


async def ingest_route(request: Request) -> JSONResponse:
    """Clone or re-sync a git repo and queue a code re-index.

    `branch` applies to the initial clone only; an existing checkout is
    fast-forwarded on its current branch. Remove and re-add to switch branch.
    """
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}", 400)
    try:
        url = cache.validate_repo_url(body.get("url"))
        name = cache.derive_repo_name(url, body.get("name"))
        branch = cache.check_branch(body.get("branch"))
    except ValueError as exc:
        return error(str(exc), 400)

    if _low_on_disk():
        return error(
            f"free disk space below the {cache.DISK_HEADROOM_BYTES} byte headroom "
            f"plus the {cache.REPO_MAX_BYTES} byte checkout cap",
            507,
        )
    try:
        key = request.state.key
        job = await job_store.admit_repo(
            job_id=uuid.uuid4().hex,
            key_id=key.key_id,
            key_label=key.label,
            name=name,
            action="ingest",
            url=url,
            branch=branch,
        )
    except job_store.BacklogFullError as exc:
        return error(str(exc), 429)
    return JSONResponse(
        {
            "job_id": job.job_id,
            "name": name,
            "status": job.status,
            "status_url": f"/repos/jobs/{job.job_id}",
        },
        status_code=202,
    )


async def remove_route(request: Request) -> JSONResponse:
    """Remove a cached repo and queue a code re-index to tear down its rows.

    Restricted to an admin key or the repo's owner (the key label that first
    ingested it); a repo with no owner record is admin-only (fail-closed).
    """
    try:
        name = cache.check_name(request.path_params["name"])
    except ValueError as exc:
        return error(str(exc), 400)
    dest = cache.CACHE_ROOT / name
    if not dest.is_dir():
        return error("repo not found", 404)
    key = request.state.key
    if not key.is_admin and cache.read_owner(name) != key.label:
        return error("only the repo owner or an admin can remove this repo", 403)
    try:
        job = await job_store.admit_repo(
            job_id=uuid.uuid4().hex,
            key_id=key.key_id,
            key_label=key.label,
            name=name,
            action="remove",
            url=None,
            branch=None,
        )
    except job_store.BacklogFullError as exc:
        return error(str(exc), 429)
    return JSONResponse(
        {
            "job_id": job.job_id,
            "name": name,
            "status": job.status,
            "status_url": f"/repos/jobs/{job.job_id}",
        },
        status_code=202,
    )


async def list_route(request: Request) -> JSONResponse:
    """List cached repositories."""
    del request
    return JSONResponse(await cache.list_repos())


async def job_route(request: Request) -> JSONResponse:
    """Return durable repo job state."""
    job = await job_store.get_job(request.path_params["job_id"], kind="repo")
    if job is None:
        return error("repo job not found", 404)
    return JSONResponse(job.response())
