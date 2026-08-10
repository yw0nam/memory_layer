"""Contract test for the tag object a release creates.

Separate from test_release.py, which stays pure: this one drives the real
git effects against a throwaway repository, because the property under test
is the kind of tag object git ends up holding.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import release


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture()
def scratch_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repo on main with one commit, wired into release.py's paths."""
    monkeypatch.delenv("CLAUDECODE", raising=False)
    repo = tmp_path / "scratch"
    (repo / "scripts").mkdir(parents=True)
    (repo / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.1.0"\n')
    _git("init", "-q", "-b", "main", ".", cwd=repo)
    _git("config", "user.email", "t@t", cwd=repo)
    _git("config", "user.name", "t", cwd=repo)
    _git("add", "-A", cwd=repo)
    _git("commit", "-qm", "feat: seed (#1)", cwd=repo)

    monkeypatch.setattr(release, "REPO_ROOT", repo)
    monkeypatch.setattr(release, "PYPROJECT", repo / "pyproject.toml")
    monkeypatch.setattr(release, "CHANGELOG", repo / "CHANGELOG.md")
    return repo


def test_release_creates_an_annotated_tag(scratch_repo: Path):
    """Lightweight tags are skipped by the push the script instructs, stranding the release."""
    assert release.main([]) == 0
    tag_sha = _git("rev-parse", "v0.1.0", cwd=scratch_repo)
    assert _git("cat-file", "-t", tag_sha, cwd=scratch_repo) == "tag"


def test_release_tag_points_at_the_release_commit(scratch_repo: Path):
    assert release.main([]) == 0
    assert _git("rev-parse", "v0.1.0^{commit}", cwd=scratch_repo) == _git(
        "rev-parse", "HEAD", cwd=scratch_repo
    )
    assert _git("log", "-1", "--format=%s", cwd=scratch_repo) == "chore: release v0.1.0"
