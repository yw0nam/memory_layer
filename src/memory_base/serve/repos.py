"""URL-driven multi-repo code ingestion: git cache management + REST routes.

Clones/pulls git repositories into the code cache and (re)runs the CocoIndex
code indexer over it, so repos can be added and removed at runtime.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import asyncpg
from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.serve import job_store
from memory_base.serve.job_store import _iso_time

REPO_MAX_QUEUED = int(os.getenv("REPO_MAX_QUEUED", "10"))
REPO_MAX_BYTES = int(os.getenv("REPO_MAX_BYTES", str(2 * 1024**3)))
DISK_HEADROOM_BYTES = int(os.getenv("REPO_DISK_HEADROOM_BYTES", str(1024**3)))
SIZE_POLL_SECONDS = 5
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


def _dir_size(path: Path) -> int:
    """Total bytes of regular files under path; a missing path is 0 bytes."""
    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.islink(file_path):
                continue
            try:
                total += os.stat(file_path).st_size
            except OSError:
                continue
    return total


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill proc's whole process group, or proc alone when it does not lead one."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    try:
        if pgid == proc.pid:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


async def _watch_size(proc: asyncio.subprocess.Process, dest: Path) -> None:
    """Kill proc once dest grows past the size cap; return once proc is not running."""
    waiter = asyncio.ensure_future(proc.wait())
    while True:
        done, _ = await asyncio.wait({waiter}, timeout=SIZE_POLL_SECONDS)
        if done:
            return
        if await asyncio.to_thread(_dir_size, dest) > REPO_MAX_BYTES:
            _kill_tree(proc)
            await waiter
            return


async def _run_git_bounded(dest: Path, *args: str, cwd: str | None = None) -> str:
    """Run a git subprocess under `dest`, guarded by the size watchdog and a final check."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    _, (out, err) = await asyncio.gather(_watch_size(proc, dest), proc.communicate())
    if await asyncio.to_thread(_dir_size, dest) > REPO_MAX_BYTES:
        raise RepoError(f"checkout exceeds the {REPO_MAX_BYTES} byte size limit")
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
    pre_existing = dest.exists()
    try:
        await _run_git_bounded(dest, *_clone_args(url, str(dest), branch))
    except RepoError:
        if not pre_existing:
            shutil.rmtree(dest, ignore_errors=True)
        raise


async def pull(dest: Path) -> None:
    is_shallow = await _run_git("-C", str(dest), "rev-parse", "--is-shallow-repository")
    if is_shallow == "true":
        await _run_git_bounded(dest, "-C", str(dest), "fetch", "--filter=blob:none", "--unshallow")
    await _run_git_bounded(dest, "-C", str(dest), "pull", "--ff-only")


def remove(dest: Path) -> None:
    shutil.rmtree(dest)


async def _repo_chunk_counts() -> dict[str, int]:
    """Chunk count per repo; empty when the table or the DB is unavailable."""
    try:
        async with db.acquire() as conn:
            rows = await conn.fetch(
                f'SELECT repo, COUNT(*) AS n FROM "{PG_SCHEMA}"."code_chunks" GROUP BY repo'
            )
        return {row["repo"]: row["n"] for row in rows}
    except (OSError, asyncpg.PostgresError):
        return {}


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

    def mark_running(self) -> None:
        self.touch(status="running")

    def mark_succeeded(self) -> None:
        self.touch(status="succeeded")

    def mark_failed(self) -> None:
        self.touch(status="failed")

    def response(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = _iso_time(self.created_at)
        payload["updated_at"] = _iso_time(self.updated_at)
        return payload


class RepoJobRegistry(job_store.JobRegistry):
    """Bounded repo-job state; serializes index runs under a semaphore."""

    def __init__(
        self,
        *,
        max_queued: int = REPO_MAX_QUEUED,
        max_concurrent: int = 1,
        ttl_seconds: int = job_store.JOB_TTL_SECONDS,
        max_completed: int = job_store.MAX_COMPLETED_JOBS,
    ) -> None:
        # ponytail: serialize index runs; coalesce only if bursts get heavy.
        super().__init__(
            kind="repo",
            job_cls=RepoJob,
            terminal_statuses=TERMINAL_STATUSES,
            queue_full_message="repo job queue is full",
            max_queued=max_queued,
            max_concurrent=max_concurrent,
            ttl_seconds=ttl_seconds,
            max_completed=max_completed,
        )

    def create(self, name: str, action: str) -> RepoJob:
        now = time.time()
        job = RepoJob(
            job_id=uuid.uuid4().hex, name=name, action=action, created_at=now, updated_at=now
        )
        self.register(job)
        return job


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


def _low_on_disk() -> bool:
    """True when the cache volume cannot hold a full-size checkout above the headroom floor.

    Walks up to the nearest existing parent when the cache dir is not yet
    created; an unreadable volume counts as low.
    """
    path = CACHE_ROOT
    while not path.exists():
        parent = path.parent
        if parent == path:
            return False
        path = parent
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return True
    return usage.free < DISK_HEADROOM_BYTES + REPO_MAX_BYTES


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
    if _low_on_disk():
        return _error(
            f"free disk space below the {DISK_HEADROOM_BYTES} byte headroom "
            f"plus the {REPO_MAX_BYTES} byte checkout cap",
            507,
        )
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
