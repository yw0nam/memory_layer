"""Contract test for the tag object a release creates.

Separate from test_release.py, which stays pure: this one drives the real
git effects against a throwaway repository, because the property under test
is the kind of tag object git ends up holding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import release
from gitcmd import git_output


@pytest.fixture()
def scratch_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repo on main with one commit, wired into release.py's paths."""
    monkeypatch.delenv("CLAUDECODE", raising=False)
    repo = tmp_path / "scratch"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1.0"\n')
    git_output("init", "-q", "-b", "main", ".", cwd=repo)
    git_output("config", "user.email", "t@t", cwd=repo)
    git_output("config", "user.name", "t", cwd=repo)
    git_output("add", "-A", cwd=repo)
    git_output("commit", "-qm", "feat: seed (#1)", cwd=repo)

    monkeypatch.setattr(release, "REPO_ROOT", repo)
    monkeypatch.setattr(release, "PYPROJECT", repo / "pyproject.toml")
    monkeypatch.setattr(release, "CHANGELOG", repo / "CHANGELOG.md")
    monkeypatch.setattr(release, "LOCKFILE", repo / "uv.lock")

    # Stub out `uv lock` alone: it would hit the network for a scratch repo with no real deps.
    real_run = release._run

    def fake_run(*args: str) -> str:
        if args == ("uv", "lock"):
            (repo / "uv.lock").write_text("")
            return ""
        return real_run(*args)

    monkeypatch.setattr(release, "_run", fake_run)
    return repo


def test_release_creates_an_annotated_tag(scratch_repo: Path):
    """Lightweight tags are skipped by the push the script instructs, stranding the release."""
    assert release.main([]) == 0
    tag_sha = git_output("rev-parse", "v0.1.0", cwd=scratch_repo)
    assert git_output("cat-file", "-t", tag_sha, cwd=scratch_repo) == "tag"


def test_release_tag_points_at_the_release_commit(scratch_repo: Path):
    assert release.main([]) == 0
    assert git_output("rev-parse", "v0.1.0^{commit}", cwd=scratch_repo) == git_output(
        "rev-parse", "HEAD", cwd=scratch_repo
    )
    assert git_output("log", "-1", "--format=%s", cwd=scratch_repo) == ("chore: release v0.1.0")
