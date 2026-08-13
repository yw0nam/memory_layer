"""Unit tests for the pure parsing/grouping/rendering/bumping logic in scripts/release.py.

No test here touches git or the filesystem: commit subjects are literal strings,
and the release script's own git/file effects are exercised manually, not here.
"""

from __future__ import annotations

import pytest

from release import (
    ParsedCommit,
    bump_version,
    bump_version_line,
    group_commits,
    is_release_commit,
    main,
    parse_commit_subject,
    render_version_section,
)

# ---- parse_commit_subject ---------------------------------------------------


def test_parses_type_and_pr_number():
    assert parse_commit_subject("feat: add search filters (#42)") == ParsedCommit(
        type="feat",
        description="add search filters",
        pr_number=42,
    )


def test_parses_subject_with_no_pr_number():
    assert parse_commit_subject("feat: deep-search surface and memory source rename") == (
        ParsedCommit(
            type="feat",
            description="deep-search surface and memory source rename",
            pr_number=None,
        )
    )


def test_trailing_parenthetical_that_is_not_a_pr_number_stays_in_the_description():
    subject = "docs: add agent skills config (issue tracker, triage labels, domain docs)"
    parsed = parse_commit_subject(subject)
    assert parsed.type == "docs"
    assert parsed.pr_number is None
    assert (
        parsed.description == "add agent skills config (issue tracker, triage labels, domain docs)"
    )


def test_unparseable_subject_keeps_full_text_and_has_no_type():
    assert parse_commit_subject("first commit") == ParsedCommit(
        type=None, description="first commit", pr_number=None
    )


# ---- group_commits -----------------------------------------------------------


def _commit(commit_type, description="x"):
    return ParsedCommit(type=commit_type, description=description, pr_number=None)


def test_groups_known_types_into_their_named_sections():
    commits = [_commit("feat"), _commit("fix"), _commit("refactor"), _commit("perf")]
    grouped = group_commits(commits)
    assert grouped == {
        "Added": [commits[0]],
        "Fixed": [commits[1]],
        "Refactored": [commits[2]],
        "Performance": [commits[3]],
    }


def test_chore_ci_build_and_style_all_collapse_into_maintenance():
    commits = [_commit("chore"), _commit("ci"), _commit("build"), _commit("style")]
    grouped = group_commits(commits)
    assert list(grouped) == ["Maintenance"]
    assert grouped["Maintenance"] == commits


def test_unparseable_type_lands_in_other():
    commits = [_commit(None, "first commit")]
    grouped = group_commits(commits)
    assert grouped == {"Other": commits}


def test_empty_sections_are_omitted():
    grouped = group_commits([_commit("feat")])
    assert list(grouped) == ["Added"]


def test_section_order_is_fixed_regardless_of_input_order():
    commits = [_commit("chore"), _commit("test"), _commit("feat"), _commit("docs")]
    grouped = group_commits(commits)
    assert list(grouped) == ["Added", "Documentation", "Tests", "Maintenance"]


def test_commit_order_within_a_section_is_preserved():
    first = _commit("feat", "first")
    second = _commit("feat", "second")
    grouped = group_commits([first, second])
    assert grouped["Added"] == [first, second]


# ---- render_version_section --------------------------------------------------


def test_render_includes_version_header_and_date():
    rendered = render_version_section("0.1.0", "2026-08-07", [])
    assert rendered.startswith("## [0.1.0] - 2026-08-07")


def test_render_appends_pr_number_when_present():
    commits = [ParsedCommit("feat", "add search filters", 42)]
    rendered = render_version_section("0.1.0", "2026-08-07", commits)
    assert "- add search filters (#42)" in rendered


def test_render_omits_pr_suffix_when_absent():
    commits = [ParsedCommit("feat", "add search filters", None)]
    rendered = render_version_section("0.1.0", "2026-08-07", commits)
    assert "- add search filters" in rendered
    assert "(#" not in rendered


def test_render_only_includes_non_empty_section_headings():
    commits = [ParsedCommit("feat", "a", None), ParsedCommit("fix", "b", None)]
    rendered = render_version_section("0.1.0", "2026-08-07", commits)
    assert "### Added" in rendered
    assert "### Fixed" in rendered
    assert "### Refactored" not in rendered
    assert "### Other" not in rendered


def test_render_puts_sections_in_fixed_order():
    commits = [
        ParsedCommit("test", "t", None),
        ParsedCommit("feat", "f", None),
        ParsedCommit(None, "unparseable", None),
    ]
    rendered = render_version_section("0.1.0", "2026-08-07", commits)
    assert rendered.index("### Added") < rendered.index("### Tests") < rendered.index("### Other")


# ---- bump_version -------------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "level", "expected"),
    [
        ("0.1.0", "major", "1.0.0"),
        ("0.1.0", "minor", "0.2.0"),
        ("0.1.0", "patch", "0.1.1"),
        ("1.4.9", "minor", "1.5.0"),
        ("1.4.9", "patch", "1.4.10"),
    ],
)
def test_bump_version(version, level, expected):
    assert bump_version(version, level) == expected


def test_bump_version_rejects_unknown_level():
    with pytest.raises(ValueError, match="major, minor, patch"):
        bump_version("0.1.0", "bogus")


# ---- bump_version_line (pyproject.toml text substitution) --------------------


def test_bump_version_line_replaces_only_the_project_version():
    text = '[project]\nname = "memory-base"\nversion = "0.1.0"\n\n[tool.ruff]\ntarget-version = "py312"\n'
    updated = bump_version_line(text, "0.2.0")
    assert 'version = "0.2.0"' in updated
    assert 'target-version = "py312"' in updated
    assert '"0.1.0"' not in updated


def test_bump_version_line_raises_when_no_version_line_exists():
    with pytest.raises(RuntimeError, match="version"):
        bump_version_line('[project]\nname = "memory-base"\n', "0.2.0")


# ---- is_release_commit --------------------------------------------------------


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("chore: release v0.1.0", True),
        ("chore: release v1.10.2", True),
        ("chore: releases are now automated", False),
        ("feat: release gating", False),
    ],
)
def test_is_release_commit(subject, expected):
    assert is_release_commit(subject) is expected


# ---- agent refusal ------------------------------------------------------------


def test_refuses_to_run_under_an_agent(monkeypatch, capsys):
    monkeypatch.setenv("CLAUDECODE", "1")
    assert main([]) == 1
    assert "worktree" in capsys.readouterr().err


def test_agent_refusal_precedes_every_git_call(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setattr("release._run", lambda *args: pytest.fail(f"ran git: {args}"))
    assert main(["minor"]) == 1


# ---- main(): uv.lock refresh --------------------------------------------------


def _stub_git(calls):
    """Fake `_run` that answers the git queries `main()` makes before its effects."""

    def fake_run(*args):
        calls.append(args)
        if args == ("git", "status", "--porcelain"):
            return ""
        if args == ("git", "branch", "--show-current"):
            return "main\n"
        if args[:2] in (("git", "tag"), ("git", "log")):
            return ""
        return ""

    return fake_run


def test_uv_lock_runs_before_git_add_with_uv_lock_staged(monkeypatch, tmp_path):
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.setattr("release.REPO_ROOT", tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "memory-base"\nversion = "0.1.0"\n', encoding="utf-8")
    monkeypatch.setattr("release.PYPROJECT", pyproject)
    monkeypatch.setattr("release.CHANGELOG", tmp_path / "CHANGELOG.md")
    monkeypatch.setattr("release.LOCKFILE", tmp_path / "uv.lock")

    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr("release._run", _stub_git(calls))

    assert main(["patch"]) == 0

    lock_index = calls.index(("uv", "lock"))
    add_index = next(i for i, c in enumerate(calls) if c[:2] == ("git", "add"))
    commit_index = next(i for i, c in enumerate(calls) if c[:2] == ("git", "commit"))
    assert lock_index < add_index < commit_index
    assert "uv.lock" in calls[add_index]


def test_uv_lock_runs_even_without_a_bump_level(monkeypatch, tmp_path):
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.setattr("release.REPO_ROOT", tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "memory-base"\nversion = "0.1.0"\n', encoding="utf-8")
    monkeypatch.setattr("release.PYPROJECT", pyproject)
    monkeypatch.setattr("release.CHANGELOG", tmp_path / "CHANGELOG.md")
    monkeypatch.setattr("release.LOCKFILE", tmp_path / "uv.lock")

    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr("release._run", _stub_git(calls))

    assert main([]) == 0

    assert ("uv", "lock") in calls
