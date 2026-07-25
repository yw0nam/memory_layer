"""URL-driven multi-repo code ingestion: git cache management + REST routes.

Clones/pulls git repositories into the code cache and (re)runs the CocoIndex
code indexer over it, so repos can be added and removed at runtime.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import asyncpg
from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.common import DB_URL, PG_SCHEMA
from memory_base.serve import job_store

REPO_MAX_QUEUED = int(os.getenv("REPO_MAX_QUEUED", "10"))
JOB_TTL_SECONDS = 24 * 60 * 60
MAX_COMPLETED_JOBS = 100
TERMINAL_STATUSES = frozenset({"succeeded", "failed"})

CACHE_ROOT = Path(
    os.getenv("REPO_CACHE", Path(__file__).resolve().parents[3] / ".repos_cache")
).resolve()
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
CODE_APP = "src/memory_base/ingest/code.py"

_SCP_LIKE = re.compile(r"^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+:.+$")
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class RepoError(RuntimeError):
    """A git or indexing operation failed."""


# ---- validation (trust boundary) ------------------------------------------


def validate_repo_url(url: Any) -> str:
    """Accept only http(s)/ssh URLs or git@host:path SSH form; no whitespace.

    Blocks git argument/option injection at the request boundary.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url is required")
    url = url.strip()
    if any(char.isspace() for char in url):
        raise ValueError("url must not contain whitespace")
    if url.startswith("-"):
        raise ValueError("url must not start with '-'")
    scheme = urlsplit(url).scheme
    if scheme in {"https", "http", "ssh"}:
        if not urlsplit(url).netloc:
            raise ValueError("url must include a host")
        return url
    if _SCP_LIKE.match(url):
        return url
    raise ValueError("url must be an http(s)/ssh URL or git@host:path SSH form")


def _check_name(name: str) -> str:
    if not name or ".." in name or "/" in name or "\\" in name or not _NAME_RE.match(name):
        raise ValueError("invalid repo name")
    return name


def derive_repo_name(url: str, name: str | None) -> str:
    """Return a safe cache directory name from an explicit name or the URL."""
    if name is None or name == "":
        base = re.split(r"[/:]", url.rstrip("/"))[-1]
        name = base[:-4] if base.endswith(".git") else base
    return _check_name(name)


def _check_branch(branch: Any) -> str | None:
    if branch is None:
        return None
    if not isinstance(branch, str) or not branch.strip():
        raise ValueError("branch must be a non-empty string")
    branch = branch.strip()
    if any(char.isspace() for char in branch) or branch.startswith("-"):
        raise ValueError("invalid branch")
    return branch


# ---- git helpers ----------------------------------------------------------


async def _run_git(*args: str, cwd: str | None = None) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise RepoError(err.decode(errors="replace").strip() or f"git {args[0]} failed")
    return out.decode(errors="replace").strip()


def _clone_args(url: str, dest: str, branch: str | None) -> list[str]:
    """Git clone argv (after the `git` executable) for a blobless, full-history clone.

    `--filter=blob:none` keeps every commit (so per-file commit times are real)
    while fetching blobs lazily; no `--depth`, which collapses all files to one
    commit time.
    """
    args = ["clone", "--filter=blob:none"]
    if branch:
        args += ["--branch", branch]
    args += ["--", url, dest]
    return args


async def clone(url: str, dest: Path, branch: str | None = None) -> None:
    await _run_git(*_clone_args(url, str(dest), branch))


async def pull(dest: Path) -> None:
    is_shallow = await _run_git("-C", str(dest), "rev-parse", "--is-shallow-repository")
    if is_shallow == "true":
        await _run_git("-C", str(dest), "fetch", "--filter=blob:none", "--unshallow")
    await _run_git("-C", str(dest), "pull", "--ff-only")


def remove(dest: Path) -> None:
    shutil.rmtree(dest)


async def _repo_chunk_counts() -> dict[str, int]:
    """Chunk count per repo; empty when the table or the DB is unavailable."""
    try:
        conn = await asyncpg.connect(DB_URL)
    except (OSError, asyncpg.PostgresError):
        return {}
    try:
        rows = await conn.fetch(
            f'SELECT repo, COUNT(*) AS n FROM "{PG_SCHEMA}"."code_chunks" GROUP BY repo'
        )
        return {row["repo"]: row["n"] for row in rows}
    except asyncpg.PostgresError:
        return {}
    finally:
        await conn.close()


async def list_repos() -> list[dict[str, Any]]:
    """List cached repos with origin URL, current branch, HEAD, and chunk counts."""
    counts = await _repo_chunk_counts()
    result: list[dict[str, Any]] = []
    if not CACHE_ROOT.exists():
        return result
    for entry in sorted(CACHE_ROOT.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        async def _git(*args: str) -> str:
            try:
                return await _run_git("-C", str(entry), *args)
            except RepoError:
                return ""

        result.append(
            {
                "name": entry.name,
                "url": await _git("remote", "get-url", "origin"),
                "branch": await _git("rev-parse", "--abbrev-ref", "HEAD"),
                "head": await _git("rev-parse", "--short", "HEAD"),
                "chunks": counts.get(entry.name, 0),
            }
        )
    return result


# ---- index runner ---------------------------------------------------------


def _index_command() -> list[str]:
    """Index argv; --no-sync keeps the runtime env from being re-resolved."""
    return ["uv", "run", "--no-sync", "cocoindex", "update", CODE_APP]


async def run_index() -> None:
    # ponytail: subprocess for isolation; move in-process if latency bites.
    proc = await asyncio.create_subprocess_exec(
        *_index_command(),
        cwd=str(PACKAGE_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RepoError(err.decode(errors="replace").strip()[-2000:] or "cocoindex update failed")


# ---- job registry ---------------------------------------------------------


@dataclass
class RepoJob:
    job_id: str
    name: str
    action: str  # "ingest" | "remove"
    status: str = "queued"
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def touch(self, *, status: str | None = None) -> None:
        if status is not None:
            self.status = status
        self.updated_at = time.time()

    def response(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = _iso_time(self.created_at)
        payload["updated_at"] = _iso_time(self.updated_at)
        return payload


def _iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


class RepoJobRegistry:
    """Bounded repo-job state; serializes index runs under a semaphore."""

    def __init__(
        self,
        *,
        max_queued: int = REPO_MAX_QUEUED,
        max_concurrent: int = 1,
        ttl_seconds: int = JOB_TTL_SECONDS,
        max_completed: int = MAX_COMPLETED_JOBS,
    ) -> None:
        # ponytail: serialize index runs; coalesce only if bursts get heavy.
        self.max_queued = max_queued
        self.ttl_seconds = ttl_seconds
        self.max_completed = max_completed
        self.jobs: OrderedDict[str, RepoJob] = OrderedDict()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task[None]] = set()

    def cleanup(self, now: float | None = None) -> None:
        # ttl_seconds sweep kept: test_repo_job_registry_queue_bound_and_ttl
        # asserts on it directly, even though Redis EXPIRE also owns job_store's copy.
        current = time.time() if now is None else now
        expired = [
            job_id
            for job_id, job in self.jobs.items()
            if job.status in TERMINAL_STATUSES and current - job.updated_at >= self.ttl_seconds
        ]
        for job_id in expired:
            self.jobs.pop(job_id, None)
        completed = [job_id for job_id, job in self.jobs.items() if job.status in TERMINAL_STATUSES]
        overflow = completed[: max(0, len(completed) - self.max_completed)]
        for job_id in overflow:
            self.jobs.pop(job_id, None)

    def queued_count(self) -> int:
        return sum(job.status == "queued" for job in self.jobs.values())

    def has_capacity(self) -> bool:
        self.cleanup()
        return self.queued_count() < self.max_queued

    def create(self, name: str, action: str) -> RepoJob:
        self.cleanup()
        if not self.has_capacity():
            raise OverflowError("repo job queue is full")
        now = time.time()
        job = RepoJob(
            job_id=uuid.uuid4().hex, name=name, action=action, created_at=now, updated_at=now
        )
        self.jobs[job.job_id] = job
        return job

    def start(self, job: RepoJob, runner: Callable[[], Awaitable[None]]) -> None:
        async def run() -> None:
            await job_store.save("repo", job)
            async with self._semaphore:
                job.touch(status="running")
                await job_store.save("repo", job)
                try:
                    await runner()
                    job.touch(status="succeeded")
                except Exception as exc:
                    job.error = str(exc) or type(exc).__name__
                    job.touch(status="failed")
                finally:
                    await job_store.save("repo", job)
                    self.cleanup()

        task = asyncio.create_task(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def get(self, job_id: str) -> RepoJob | None:
        self.cleanup()
        job = self.jobs.get(job_id)
        if job is not None:
            return job
        return await job_store.load("repo", job_id, RepoJob, TERMINAL_STATUSES)


registry = RepoJobRegistry()


# ---- job runners ----------------------------------------------------------


async def _run_ingest_job(url: str, dest: Path, branch: str | None) -> None:
    if dest.exists():
        await pull(dest)
    else:
        await clone(url, dest, branch)
    await run_index()


async def _run_remove_job(dest: Path) -> None:
    remove(dest)
    await run_index()


# ---- routes ---------------------------------------------------------------


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


async def _json_body(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


async def ingest_repo_route(request: Request) -> JSONResponse:
    """Clone or re-sync a git repo and queue a code re-index.

    `branch` applies to the initial clone only; an existing checkout is
    fast-forwarded on its current branch. Remove and re-add to switch branch.
    """
    try:
        body = await _json_body(request)
    except Exception as exc:
        return _error(f"invalid JSON body: {exc}", 400)
    try:
        url = validate_repo_url(body.get("url"))
        name = derive_repo_name(url, body.get("name"))
        branch = _check_branch(body.get("branch"))
    except ValueError as exc:
        return _error(str(exc), 400)

    if not registry.has_capacity():
        return _error("repo job queue is full", 429)
    dest = CACHE_ROOT / name
    try:
        job = registry.create(name, "ingest")
    except OverflowError as exc:
        return _error(str(exc), 429)
    registry.start(job, lambda: _run_ingest_job(url, dest, branch))
    return JSONResponse(
        {
            "job_id": job.job_id,
            "name": name,
            "status": job.status,
            "status_url": f"/repos/jobs/{job.job_id}",
        },
        status_code=202,
    )


async def remove_repo_route(request: Request) -> JSONResponse:
    """Remove a cached repo and queue a code re-index to tear down its rows."""
    try:
        name = _check_name(request.path_params["name"])
    except ValueError as exc:
        return _error(str(exc), 400)
    dest = CACHE_ROOT / name
    if not dest.is_dir():
        return _error("repo not found", 404)
    if not registry.has_capacity():
        return _error("repo job queue is full", 429)
    try:
        job = registry.create(name, "remove")
    except OverflowError as exc:
        return _error(str(exc), 429)
    registry.start(job, lambda: _run_remove_job(dest))
    return JSONResponse(
        {
            "job_id": job.job_id,
            "name": name,
            "status": job.status,
            "status_url": f"/repos/jobs/{job.job_id}",
        },
        status_code=202,
    )


async def list_repos_route(request: Request) -> JSONResponse:
    """List cached repositories."""
    del request
    return JSONResponse(await list_repos())


async def repo_job_route(request: Request) -> JSONResponse:
    """Return retained repo job state."""
    job = await registry.get(request.path_params["job_id"])
    if job is None:
        return _error("repo job not found", 404)
    return JSONResponse(job.response())
