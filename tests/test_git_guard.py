"""Contract tests for the PreToolUse git guard.

The guard is a shell script, so it is exercised the way the harness calls it:
a hook payload on stdin, a permission decision on stdout.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gitcmd import git_output

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / ".claude" / "hooks" / "pretool-bash-guard.sh"


def primary_worktree() -> Path:
    """This project's primary checkout, shared by every worktree of it."""
    return Path(
        git_output("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=REPO_ROOT)
    ).parent


on_main = pytest.mark.skipif(
    git_output("branch", "--show-current", cwd=primary_worktree()) != "main",
    reason="the primary checkout is not on main",
)


def run_guard(command: str, cwd: Path, project_dir: Path | None = REPO_ROOT) -> dict:
    """Invoke the guard with a hook payload; return its decision, or {} when it allows."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin"}
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    result = subprocess.run(
        ["bash", str(GUARD)],
        input=json.dumps({"tool_input": {"command": command}, "cwd": str(cwd)}),
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout) if result.stdout.strip() else {}


def denied(decision: dict) -> bool:
    return decision.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"


@pytest.fixture()
def foreign_repo(tmp_path: Path) -> Path:
    """A repository that is not this project, on a branch named main."""
    repo = tmp_path / "foreign"
    repo.mkdir()
    for args in (
        ["git", "init", "-q", "-b", "main", "."],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "commit", "-q", "--allow-empty", "-m", "seed"],
    ):
        subprocess.run(args, cwd=repo, check=True, capture_output=True)
    return repo


@on_main
def test_denies_commit_on_this_repo_main():
    assert denied(run_guard("git commit -m x", cwd=primary_worktree()))


@on_main
def test_denies_push_on_this_repo_main():
    assert denied(run_guard("git push", cwd=primary_worktree()))


def test_allows_commit_in_a_foreign_repo_on_main(foreign_repo: Path):
    assert not denied(run_guard("git commit -m x", cwd=foreign_repo))


def test_allows_push_in_a_foreign_repo_on_main(foreign_repo: Path):
    assert not denied(run_guard("git push", cwd=foreign_repo))


def test_denies_foreign_repo_when_the_project_cannot_be_identified(foreign_repo: Path):
    """An unidentifiable project keeps the strict behavior rather than opening up."""
    assert denied(run_guard("git commit -m x", cwd=foreign_repo, project_dir=None))


def test_allows_commit_on_a_branch_other_than_main(tmp_path: Path):
    worktree = tmp_path / "wt"
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "guard-probe", str(worktree), "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    try:
        assert not denied(run_guard("git commit -m x", cwd=worktree))
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "branch", "-D", "guard-probe"], cwd=REPO_ROOT, check=True, capture_output=True
        )


def test_still_blocks_reading_dotenv():
    assert denied(run_guard("cat .env", cwd=REPO_ROOT))


# ---- prose in quoted arguments is not a command -----------------------------

# Split so this module is not itself matched when a tool reads it back.
VERB = "git " + "push"


@on_main
def test_allows_a_single_line_quoted_argument_documenting_a_git_verb():
    assert not denied(
        run_guard(f'gh issue create --body "run {VERB} to publish"', cwd=primary_worktree())
    )


@on_main
def test_allows_a_heredoc_body_documenting_a_git_verb():
    """Issue and PR bodies are written as heredocs, and they document git commands."""
    command = f'gh issue create --body "$(cat <<EOF\nrun {VERB} to publish\nEOF\n)"'
    assert not denied(run_guard(command, cwd=primary_worktree()))


@on_main
def test_allows_a_multi_line_quoted_argument_documenting_a_git_verb():
    assert not denied(
        run_guard(
            f'gh pr create --body "first line\n{VERB} --follow-tags\n"', cwd=primary_worktree()
        )
    )


@on_main
def test_still_denies_a_real_command_following_a_heredoc():
    """Stripping quoted prose must not hide a real command elsewhere in the line."""
    command = f'gh issue create --body "$(cat <<EOF\ndocs\nEOF\n)" && {VERB}'
    assert denied(run_guard(command, cwd=primary_worktree()))


@on_main
def test_still_denies_a_real_command_between_apostrophes_on_separate_lines():
    """Apostrophe stripping stays line-scoped, so a pair of them cannot span a real command."""
    command = f'echo "it\'s fine"\n{VERB}\necho "that\'s all"'
    assert denied(run_guard(command, cwd=primary_worktree()))
