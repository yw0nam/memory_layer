"""A repo URL is caller-supplied, so a checkout must be bounded.

`POST /repos` clones an arbitrary remote into the cache volume that also holds
the CocoIndex ledger. Without a cap, one oversized repo fills the volume and
takes indexing down with it.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time

import httpx
import pytest

from memory_base.serve import api, repos


def _post(path, **kwargs):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.post(path, **kwargs)

    return asyncio.run(request())


class AcceptingRegistry:
    """Accepts every job so a rejection can only come from the guard."""

    def __init__(self):
        self.job = None
        self.runner = None

    def has_capacity(self):
        return True

    def create(self, name, action):
        now = time.time()
        self.job = repos.RepoJob("job-1", name, action, created_at=now, updated_at=now)
        return self.job

    def start(self, job, runner):
        self.runner = runner

    async def get(self, job_id):
        return self.job


def _make_git_repo(path, payload_bytes=0):
    path.mkdir(parents=True, exist_ok=True)
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    (path / "main.py").write_text("print('hi')\n")
    if payload_bytes:
        (path / "payload.txt").write_bytes(b"x" * payload_bytes)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True, env=env)


# ---- measuring a checkout --------------------------------------------------


def test_dir_size_sums_nested_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one").write_bytes(b"x" * 1000)
    (tmp_path / "a" / "b").mkdir()
    (tmp_path / "a" / "b" / "two").write_bytes(b"x" * 2000)

    assert repos._dir_size(tmp_path) == 3000


def test_dir_size_of_a_missing_path_is_zero(tmp_path):
    assert repos._dir_size(tmp_path / "gone") == 0


def test_dir_size_does_not_follow_symlinks(tmp_path):
    """A repo can ship a symlink loop; measuring must still terminate."""
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "file").write_bytes(b"x" * 500)
    (tmp_path / "real" / "loop").symlink_to(tmp_path / "real", target_is_directory=True)

    assert repos._dir_size(tmp_path) == 500


# ---- bounding the clone itself ---------------------------------------------


def test_watchdog_kills_a_process_that_outgrows_the_cap(tmp_path, monkeypatch):
    """A huge remote must die mid-clone, not after it has filled the disk."""
    dest = tmp_path / "checkout"
    dest.mkdir()
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 64 * 1024)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 0.02)

    grower = (
        "import pathlib, time\n"
        "blob = pathlib.Path('blob')\n"
        "for i in range(400):\n"
        "    blob.write_bytes(b'x' * 8192 * (i + 1))\n"
        "    time.sleep(0.01)\n"
    )

    async def scenario():
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            grower,
            cwd=str(dest),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await repos._watch_size(proc, dest)
        return await proc.wait()

    started = time.monotonic()
    returncode = asyncio.run(asyncio.wait_for(scenario(), 20))

    assert returncode != 0
    assert time.monotonic() - started < 3


def test_clone_rejects_a_checkout_that_lands_over_the_cap(tmp_path, monkeypatch):
    """The final check catches what the poll interval was too coarse to see."""
    origin = tmp_path / "origin"
    _make_git_repo(origin, payload_bytes=300_000)
    dest = tmp_path / "cache" / "repo"
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 50_000)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 30)

    with pytest.raises(repos.RepoError):
        asyncio.run(repos.clone(str(origin), dest))


def test_a_rejected_clone_leaves_no_partial_checkout(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    _make_git_repo(origin, payload_bytes=300_000)
    dest = tmp_path / "cache" / "repo"
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 50_000)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 30)

    with pytest.raises(repos.RepoError):
        asyncio.run(repos.clone(str(origin), dest))

    assert not dest.exists()


def test_a_clone_within_the_cap_is_untouched(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    _make_git_repo(origin, payload_bytes=1000)
    dest = tmp_path / "cache" / "repo"
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 0.02)

    asyncio.run(repos.clone(str(origin), dest))

    assert (dest / "main.py").read_text() == "print('hi')\n"


def test_pull_is_bounded_too(tmp_path, monkeypatch):
    """A repo that grows past the cap after the fact must not be re-synced in."""
    origin = tmp_path / "origin"
    _make_git_repo(origin, payload_bytes=1000)
    dest = tmp_path / "cache" / "repo"
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 30)
    asyncio.run(repos.clone(str(origin), dest))

    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 100)
    with pytest.raises(repos.RepoError):
        asyncio.run(repos.pull(dest))


# ---- refusing before the disk is gone --------------------------------------


def test_post_repos_refuses_when_free_space_is_below_the_headroom(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(repos, "DISK_HEADROOM_BYTES", 10 * 1024**3)
    monkeypatch.setattr(
        repos.shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(100, 90, 10)
    )
    registry = AcceptingRegistry()
    monkeypatch.setattr(repos, "registry", registry)

    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})

    assert response.status_code == 507
    assert registry.job is None


def test_post_repos_proceeds_when_the_disk_has_room(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(repos, "DISK_HEADROOM_BYTES", 1024)
    monkeypatch.setattr(
        repos.shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(100, 0, 100)
    )
    registry = AcceptingRegistry()
    monkeypatch.setattr(repos, "registry", registry)

    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})

    assert response.status_code == 202
    assert registry.job is not None


def test_an_absent_cache_root_does_not_block_ingestion(monkeypatch, tmp_path):
    """The cache dir is created on first use; its absence is not a full disk."""
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path / "not-created-yet")
    registry = AcceptingRegistry()
    monkeypatch.setattr(repos, "registry", registry)

    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})

    assert response.status_code == 202
