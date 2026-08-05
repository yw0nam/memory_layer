"""A repo URL is caller-supplied, so a checkout must be bounded.

`POST /repos` clones an arbitrary remote into the cache volume that also holds
the CocoIndex ledger. Without a cap, one oversized repo fills the volume and
takes indexing down with it.
"""

from __future__ import annotations

import asyncio
import os
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
            headers={"X-API-Key": "test-key"},
        ) as client:
            return await client.post(path, **kwargs)

    return asyncio.run(request())


class AcceptingBacklog:
    """Accepts every job so a rejection can only come from the guard."""

    def __init__(self):
        self.job = None

    async def admit(self, **kwargs):
        now = time.time()
        self.job = repos.RepoJob(
            job_id="job-1",
            name=kwargs["name"],
            action=kwargs["action"],
            url=kwargs["url"],
            branch=kwargs["branch"],
            key_id=kwargs["key_id"],
            key_label=kwargs["key_label"],
            created_at=now,
            updated_at=now,
        )
        return self.job


_COMMITTER = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _make_git_repo(path, payload_bytes=0):
    path.mkdir(parents=True, exist_ok=True)
    env = _COMMITTER
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


def test_the_kill_takes_descendants_with_it(tmp_path, monkeypatch):
    """git spawns helpers (ssh, git-remote-*, index-pack); killing only the parent
    leaves them writing into the checkout."""
    dest = tmp_path / "checkout"
    dest.mkdir()
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 64 * 1024)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 0.02)

    parent = (
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c',\n"
        '    "import pathlib, time\\n"\n'
        "    \"blob = pathlib.Path('child-blob')\\n\"\n"
        '    "for i in range(600):\\n"\n'
        "    \"    blob.write_bytes(b'x' * 8192 * (i + 1))\\n\"\n"
        '    "    time.sleep(0.01)\\n"])\n'
        "pathlib.Path('child.pid').write_text(str(child.pid))\n"
        "time.sleep(60)\n"
    )

    async def scenario():
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            parent,
            cwd=str(dest),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        await repos._watch_size(proc, dest)
        await proc.wait()
        return int((dest / "child.pid").read_text())

    child_pid = asyncio.run(asyncio.wait_for(scenario(), 20))

    time.sleep(0.5)
    with pytest.raises(OSError):
        os.kill(child_pid, 0)


def test_git_runs_in_its_own_process_group(tmp_path, monkeypatch):
    """Killing a group only works if git leads one — otherwise it is the API's own."""
    origin = tmp_path / "origin"
    _make_git_repo(origin, payload_bytes=1000)
    dest = tmp_path / "cache" / "repo"
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 30)

    seen = []
    spawn = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        seen.append(kwargs.get("start_new_session"))
        return await spawn(*args, **kwargs)

    monkeypatch.setattr(repos.asyncio, "create_subprocess_exec", spy)
    asyncio.run(repos.clone(str(origin), dest))

    assert seen and all(started is True for started in seen)


def test_measuring_the_checkout_does_not_block_the_event_loop(tmp_path, monkeypatch):
    """A million-file checkout takes seconds to walk; the API cannot stop serving for it."""
    dest = tmp_path / "checkout"
    dest.mkdir()
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 10**12)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(repos, "_dir_size", lambda path: time.sleep(0.3) or 0)

    async def scenario():
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(1)",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        await repos._watch_size(proc, dest)
        beat.cancel()
        return ticks

    ticks = asyncio.run(asyncio.wait_for(scenario(), 20))

    assert ticks > 20


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


def test_a_failed_clone_does_not_delete_a_checkout_it_did_not_create(tmp_path, monkeypatch):
    """clone() may only remove what it wrote; an existing checkout is not its to drop."""
    origin = tmp_path / "origin"
    _make_git_repo(origin, payload_bytes=1000)
    dest = tmp_path / "cache" / "repo"
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 30)
    asyncio.run(repos.clone(str(origin), dest))

    with pytest.raises(repos.RepoError):
        asyncio.run(repos.clone(str(origin), dest))

    assert (dest / "main.py").exists()


def test_a_clone_within_the_cap_is_untouched(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    _make_git_repo(origin, payload_bytes=1000)
    dest = tmp_path / "cache" / "repo"
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 0.02)

    asyncio.run(repos.clone(str(origin), dest))

    assert (dest / "main.py").read_text() == "print('hi')\n"


def test_pull_is_bounded_too(tmp_path, monkeypatch):
    """Growth arriving through a pull is caught, not just a checkout already over the cap."""
    origin = tmp_path / "origin"
    _make_git_repo(origin, payload_bytes=1000)
    dest = tmp_path / "cache" / "repo"
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 0.02)
    asyncio.run(repos.clone(str(origin), dest))

    settled = repos._dir_size(dest)
    (origin / "payload.txt").write_bytes(b"y" * 8 * 1024 * 1024)
    subprocess.run(["git", "-C", str(origin), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(origin), "commit", "-q", "-m", "grow"],
        check=True,
        env=_COMMITTER,
    )

    # under the cap before the pull, over it once the new commit lands
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", settled + 2 * 1024 * 1024)
    with pytest.raises(repos.RepoError):
        asyncio.run(repos.pull(dest))


def test_a_rejected_pull_keeps_the_existing_checkout(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    _make_git_repo(origin, payload_bytes=1000)
    dest = tmp_path / "cache" / "repo"
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 50 * 1024 * 1024)
    monkeypatch.setattr(repos, "SIZE_POLL_SECONDS", 30)
    asyncio.run(repos.clone(str(origin), dest))

    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 100)
    with pytest.raises(repos.RepoError):
        asyncio.run(repos.pull(dest))

    assert (dest / "main.py").exists()


# ---- refusing before the disk is gone --------------------------------------


def _fake_usage(monkeypatch, *, total, free):
    usage = shutil._ntuple_diskusage(total, total - free, free)
    monkeypatch.setattr(repos.shutil, "disk_usage", lambda path: usage)


def test_post_repos_refuses_when_free_space_is_below_the_headroom(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(repos, "DISK_HEADROOM_BYTES", 1024**3)
    _fake_usage(monkeypatch, total=100 * 1024**3, free=200 * 1024**2)
    backlog = AcceptingBacklog()
    monkeypatch.setattr(repos.job_store, "admit_repo", backlog.admit)

    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})

    assert response.status_code == 507
    assert backlog.job is None


def test_an_empty_volume_smaller_than_the_headroom_still_refuses(monkeypatch, tmp_path):
    """Being wholly unused does not make a volume big enough to hold a checkout."""
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(repos, "DISK_HEADROOM_BYTES", 1024**3)
    _fake_usage(monkeypatch, total=512 * 1024**2, free=512 * 1024**2)
    backlog = AcceptingBacklog()
    monkeypatch.setattr(repos.job_store, "admit_repo", backlog.admit)

    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})

    assert response.status_code == 507
    assert backlog.job is None


def test_admission_requires_room_for_a_full_size_checkout(monkeypatch, tmp_path):
    """A cap larger than the free space cannot bound anything — refuse at the door."""
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(repos, "DISK_HEADROOM_BYTES", 1024**3)
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 2 * 1024**3)
    _fake_usage(monkeypatch, total=100 * 1024**3, free=3 * 1024**3 - 1)
    backlog = AcceptingBacklog()
    monkeypatch.setattr(repos.job_store, "admit_repo", backlog.admit)

    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})

    assert response.status_code == 507
    assert backlog.job is None


def test_an_unreadable_volume_refuses_rather_than_admits(monkeypatch, tmp_path):
    """Failing open puts the guard off exactly when the volume is in trouble."""
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)

    def unreadable(path):
        raise OSError("EIO")

    monkeypatch.setattr(repos.shutil, "disk_usage", unreadable)
    backlog = AcceptingBacklog()
    monkeypatch.setattr(repos.job_store, "admit_repo", backlog.admit)

    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})

    assert response.status_code == 507
    assert backlog.job is None


def test_post_repos_proceeds_when_the_disk_has_room(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(repos, "DISK_HEADROOM_BYTES", 1024**3)
    monkeypatch.setattr(repos, "REPO_MAX_BYTES", 2 * 1024**3)
    _fake_usage(monkeypatch, total=100 * 1024**3, free=50 * 1024**3)
    backlog = AcceptingBacklog()
    monkeypatch.setattr(repos.job_store, "admit_repo", backlog.admit)

    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})

    assert response.status_code == 202
    assert backlog.job is not None


def test_an_absent_cache_root_does_not_block_ingestion(monkeypatch, tmp_path):
    """The cache dir is created on first use; its absence is not a full disk."""
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path / "not-created-yet")
    _fake_usage(monkeypatch, total=100 * 1024**3, free=50 * 1024**3)
    backlog = AcceptingBacklog()
    monkeypatch.setattr(repos.job_store, "admit_repo", backlog.admit)

    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})

    assert response.status_code == 202
