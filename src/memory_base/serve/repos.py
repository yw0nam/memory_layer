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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

import asyncpg
from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.serve import job_store
from memory_base.serve.http import error, json_body
from memory_base.serve.job_store import JobBase

REPO_MAX_BYTES = int(os.getenv("REPO_MAX_BYTES", str(2 * 1024**3)))
DISK_HEADROOM_BYTES = int(os.getenv("REPO_DISK_HEADROOM_BYTES", str(1024**3)))
SIZE_POLL_SECONDS = 5
CACHE_ROOT = Path(
    os.getenv("REPO_CACHE", Path(__file__).resolve().parents[3] / ".repos_cache")
).resolve()
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
CODE_APP = "src/memory_base/ingest/code.py"

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class RepoError(RuntimeError):
    """A git or indexing operation failed."""


# ---- validation (trust boundary) ------------------------------------------


def validate_repo_url(url: Any) -> str:
    """Accept only http(s) URLs without userinfo or whitespace.

    Blocks git argument/option injection at the request boundary, and keeps
    credentials out of remotes: private clones authenticate through the git
    credential store, never through the URL.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url is required")
    url = url.strip()
    if any(char.isspace() for char in url):
        raise ValueError("url must not contain whitespace")
    if url.startswith("-"):
        raise ValueError("url must not start with '-'")
    parts = urlsplit(url)
    if parts.scheme not in {"https", "http"}:
        raise ValueError("url must be an http(s) URL")
    if not parts.netloc:
        raise ValueError("url must include a host")
    if parts.username is not None:
        raise ValueError("url must not embed credentials")
    return url


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
    shutil.rmtree(dest, ignore_errors=True)


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
    """List cached repos with origin URL, current branch, HEAD, chunk count, and owner."""
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
                "owner": _read_owner(entry.name),
            }
        )
    return result


# ---- ownership --------------------------------------------------------------


def _owner_path(name: str) -> Path:
    """Sidecar path recording a repo's owning key label; CACHE_ROOT is read live for tests."""
    return CACHE_ROOT / ".owners" / name


def _read_owner(name: str) -> str | None:
    try:
        return _owner_path(name).read_text().strip()
    except FileNotFoundError:
        return None


def _record_owner_if_absent(name: str, label: str) -> None:
    """Fix ownership to the first successful ingest; re-ingest never overwrites it."""
    path = _owner_path(name)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(label)


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


# ---- durable job model ----------------------------------------------------


@dataclass
class RepoJob(JobBase):
    name: str
    action: str
    url: str | None = None
    branch: str | None = None
    key_id: str = ""
    key_label: str = ""

    RESPONSE_EXCLUDE: ClassVar[frozenset[str]] = frozenset({"url", "branch", "key_id", "key_label"})

    @property
    def kind(self) -> str:
        return "repo"


# ---- job runners ----------------------------------------------------------


async def _run_ingest_job(url: str, dest: Path, branch: str | None, key_label: str) -> None:
    if dest.exists():
        try:
            valid = (dest / ".git").exists() and bool(
                await _run_git("-C", str(dest), "rev-parse", "--git-dir")
            )
        except RepoError:
            valid = False
        if valid:
            await pull(dest)
        else:
            shutil.rmtree(dest, ignore_errors=True)
            await clone(url, dest, branch)
    else:
        await clone(url, dest, branch)
    _record_owner_if_absent(dest.name, key_label)
    await run_index()


async def _run_remove_job(dest: Path) -> None:
    remove(dest)
    _owner_path(dest.name).unlink(missing_ok=True)
    await run_index()


# ---- routes ---------------------------------------------------------------


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


async def ingest_repo_route(request: Request) -> JSONResponse:
    """Clone or re-sync a git repo and queue a code re-index.

    `branch` applies to the initial clone only; an existing checkout is
    fast-forwarded on its current branch. Remove and re-add to switch branch.
    """
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}", 400)
    try:
        url = validate_repo_url(body.get("url"))
        name = derive_repo_name(url, body.get("name"))
        branch = _check_branch(body.get("branch"))
    except ValueError as exc:
        return error(str(exc), 400)

    if _low_on_disk():
        return error(
            f"free disk space below the {DISK_HEADROOM_BYTES} byte headroom "
            f"plus the {REPO_MAX_BYTES} byte checkout cap",
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


async def remove_repo_route(request: Request) -> JSONResponse:
    """Remove a cached repo and queue a code re-index to tear down its rows.

    Restricted to an admin key or the repo's owner (the key label that first
    ingested it); a repo with no owner record is admin-only (fail-closed).
    """
    try:
        name = _check_name(request.path_params["name"])
    except ValueError as exc:
        return error(str(exc), 400)
    dest = CACHE_ROOT / name
    if not dest.is_dir():
        return error("repo not found", 404)
    key = request.state.key
    if not key.is_admin and _read_owner(name) != key.label:
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


async def list_repos_route(request: Request) -> JSONResponse:
    """List cached repositories."""
    del request
    return JSONResponse(await list_repos())


async def repo_job_route(request: Request) -> JSONResponse:
    """Return durable repo job state."""
    job = await job_store.get_job(request.path_params["job_id"], kind="repo")
    if job is None:
        return error("repo job not found", 404)
    return JSONResponse(job.response())
