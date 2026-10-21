"""scripts/worktree.sh: one command creates a ready worktree with .env linked."""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "worktree.sh"
GIT_ENV = {
    **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(cwd: pathlib.Path, *args: str, check: bool = True):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=GIT_ENV,
        capture_output=True,
        text=True,
        check=check,
    )


def _run(cwd: pathlib.Path, *args: str):
    return subprocess.run(
        [str(SCRIPT), *args], cwd=cwd, env=GIT_ENV, capture_output=True, text=True
    )


@pytest.fixture
def repo(tmp_path):
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "-b", "main", "--bare", str(origin))
    repo = tmp_path / "repo"
    _git(tmp_path, "clone", "-q", str(origin), str(repo))
    (repo / ".gitignore").write_text(".env\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "push", "-q", "-u", "origin", "main")
    (repo / ".env").write_text("TOKEN=test\n")
    return repo


def test_creates_worktree_from_origin_main_with_env_link(repo, tmp_path):
    result = _run(repo, "feat/x")
    assert result.returncode == 0, result.stderr
    worktree = tmp_path / "repo.wt" / "feat-x"
    assert result.stdout.strip().splitlines()[-1] == str(worktree)
    assert _git(worktree, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "feat/x"
    assert (
        _git(worktree, "rev-parse", "HEAD").stdout == _git(repo, "rev-parse", "origin/main").stdout
    )
    assert (worktree / ".env").is_symlink()
    assert (worktree / ".env").resolve() == (repo / ".env").resolve()
    upstream = _git(worktree, "rev-parse", "--abbrev-ref", "feat/x@{upstream}", check=False)
    assert upstream.returncode != 0


def test_runs_from_inside_a_worktree(repo, tmp_path):
    assert _run(repo, "feat/x").returncode == 0
    worktree = tmp_path / "repo.wt" / "feat-x"
    result = _run(worktree, "feat/y")
    assert result.returncode == 0, result.stderr
    nested = tmp_path / "repo.wt" / "feat-y"
    assert result.stdout.strip().splitlines()[-1] == str(nested)
    assert (nested / ".env").resolve() == (repo / ".env").resolve()


def test_requires_a_branch_argument(repo):
    result = _run(repo)
    assert result.returncode == 2
    assert "usage" in result.stderr
