"""Commit-time sourcing for code chunks: real history beats checkout time."""

from __future__ import annotations

import asyncio
import os
import subprocess

from memory_base.ingest import code
from memory_base.serve import repos

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}
OLD = 1_500_000_000
NEWER = 1_600_000_000


def _commit(path, name, body, when):
    (path / name).write_text(body)
    subprocess.run(["git", "-C", str(path), "add", name], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", f"add {name}"],
        check=True,
        env={
            **os.environ,
            **_GIT_ENV,
            "GIT_AUTHOR_DATE": f"@{when} +0000",
            "GIT_COMMITTER_DATE": f"@{when} +0000",
        },
    )


def _repo_with_history(path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    _commit(path, "old.py", "def old():\n    pass\n", OLD)
    _commit(path, "new.py", "def new():\n    pass\n", NEWER)


def test_commit_time_uses_per_file_history_not_checkout_time(tmp_path):
    repo = tmp_path / "repo"
    _repo_with_history(repo)

    assert asyncio.run(code._commit_time(repo / "old.py")) == OLD
    assert asyncio.run(code._commit_time(repo / "new.py")) == NEWER


def test_commit_time_falls_back_to_file_mtime_when_git_has_nothing(tmp_path):
    repo = tmp_path / "repo"
    _repo_with_history(repo)
    untracked = repo / "untracked.py"
    untracked.write_text("x = 1\n")

    assert asyncio.run(code._commit_time(untracked)) == untracked.stat().st_mtime

    loose = tmp_path / "loose.py"
    loose.write_text("y = 2\n")
    assert asyncio.run(code._commit_time(loose)) == loose.stat().st_mtime


def test_clone_keeps_full_history_so_commit_times_exist():
    args = repos._clone_args("https://example.com/x.git", "/dest", None)
    assert "--filter=blob:none" in args
    assert "--depth" not in args, "a shallow clone collapses every file to one commit time"


def test_clone_still_honours_branch():
    args = repos._clone_args("https://example.com/x.git", "/dest", "dev")
    assert args[args.index("--branch") + 1] == "dev"
    assert args[-2:] == ["https://example.com/x.git", "/dest"]
