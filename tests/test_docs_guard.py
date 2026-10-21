"""scripts/ci/docs-guard.sh: flags change-narrative lines in stdin text and in a PR diff."""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / "scripts" / "ci" / "docs-guard.sh"
HOOK = REPO_ROOT / ".claude" / "hooks" / "posttool-edit-guard.sh"

FLAGGED = "It no longer retries."


def run_text(*lines: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(GUARD)],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
    )


def test_text_mode_flags_a_no_longer_line():
    result = run_text("The cache expires after five minutes.", FLAGGED)
    assert result.returncode == 1
    assert FLAGGED in result.stdout


def test_text_mode_flags_the_korean_vocabulary():
    result = run_text("이 기능은 제거했다.")
    assert result.returncode == 1
    assert "제거했다" in result.stdout


def test_text_mode_passes_current_state_text():
    result = run_text("The cache expires after five minutes.", "A second clean line.")
    assert result.returncode == 0, result.stdout


def test_text_mode_skips_a_line_quoting_the_vocabulary():
    result = run_text("The skip list is 제거/대체/축소, and the phrase no longer appears.")
    assert result.returncode == 0, result.stdout


# ---- diff mode --------------------------------------------------------------


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture()
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A throwaway repo: base commit holds the flagged line, branch edits docs."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "-c", "init.defaultBranch=main", "init", "-q", ".")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "docs").mkdir()
    (repo / "docs" / "a.md").write_text(f"seed line\n{FLAGGED}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "topic")
    return repo


def _commit(repo: pathlib.Path, **files: str) -> None:
    """Append each file's text to the working tree and commit it."""
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.read_text() + text if path.exists() else text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "topic")


def run_diff(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(GUARD), *args], cwd=repo, capture_output=True, text=True)


def test_diff_mode_flags_an_edited_doc(repo):
    _commit(repo, **{"docs/a.md": FLAGGED + "\n"})
    result = run_diff(repo, "main")
    assert result.returncode == 1
    assert "docs/a.md" in result.stdout


def test_diff_mode_flags_a_new_doc(repo):
    _commit(repo, **{"docs/new.md": FLAGGED + "\n"})
    result = run_diff(repo, "main")
    assert result.returncode == 1
    assert "docs/new.md" in result.stdout


def test_diff_mode_strips_one_plus_prefix(repo):
    _commit(repo, **{"docs/plus.md": "++ " + FLAGGED + "\n"})
    result = run_diff(repo, "main")
    assert result.returncode == 1
    assert f"docs/plus.md: ++ {FLAGGED}" in result.stdout


def test_diff_mode_ignores_removed_lines(repo):
    (repo / "docs" / "a.md").write_text("seed line\n")
    _git(repo, "commit", "-qam", "topic")
    result = run_diff(repo, "main")
    assert result.returncode == 0, result.stdout


def test_diff_mode_ignores_the_changelog_at_any_depth(repo):
    _commit(repo, **{"CHANGELOG.md": FLAGGED + "\n", "sub/CHANGELOG.md": FLAGGED + "\n"})
    result = run_diff(repo, "main")
    assert result.returncode == 0, result.stdout


def test_diff_mode_ignores_non_markdown(repo):
    _commit(repo, **{"notes.txt": FLAGGED + "\n"})
    result = run_diff(repo, "main")
    assert result.returncode == 0, result.stdout


def test_diff_mode_names_a_non_ascii_path_unquoted(repo):
    _commit(repo, **{"docs/한글.md": "이 기능은 제거했다.\n"})
    result = run_diff(repo, "main")
    assert result.returncode == 1
    assert "docs/한글.md" in result.stdout


def test_diff_mode_exits_2_on_an_unknown_base_ref(repo):
    result = run_diff(repo, "no-such-ref")
    assert result.returncode == 2
    assert result.stderr != ""


# ---- the PostToolUse hook delegates to the guard ----------------------------


def run_hook(project_dir: pathlib.Path, content: str) -> subprocess.CompletedProcess:
    payload = json.dumps(
        {"tool_input": {"file_path": str(project_dir / "docs" / "x.md"), "content": content}}
    )
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "CLAUDE_PROJECT_DIR": str(project_dir)}
    return subprocess.run(
        ["bash", str(HOOK)], input=payload, capture_output=True, text=True, env=env
    )


def test_hook_blocks_flagged_content(tmp_path):
    result = run_hook(tmp_path, FLAGGED)
    decision = json.loads(result.stdout)
    assert decision["decision"] == "block"
    assert FLAGGED in decision["reason"]


def test_hook_allows_clean_content(tmp_path):
    result = run_hook(tmp_path, "The cache expires after five minutes.")
    assert result.stdout.strip() == ""
