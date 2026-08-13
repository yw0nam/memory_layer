#!/usr/bin/env python3
"""Cut a release: bump the version, regenerate CHANGELOG.md, commit, and tag.

Usage: release.py [major|minor|patch]

With no argument, releases the current pyproject.toml version as-is (the
initial release). With a bump level, raises the version first. Either way,
CHANGELOG.md is regenerated in full from tags plus git log, since it is
derived state and never hand-edited.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
LOCKFILE = REPO_ROOT / "uv.lock"

BUMP_LEVELS = ("major", "minor", "patch")

TYPE_SECTIONS = {
    "feat": "Added",
    "fix": "Fixed",
    "refactor": "Refactored",
    "perf": "Performance",
    "docs": "Documentation",
    "test": "Tests",
    "chore": "Maintenance",
    "ci": "Maintenance",
    "build": "Maintenance",
    "style": "Maintenance",
}
SECTION_ORDER = (
    "Added",
    "Fixed",
    "Refactored",
    "Performance",
    "Documentation",
    "Tests",
    "Maintenance",
    "Other",
)

COMMIT_RE = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]+\))?!?:\s+(?P<description>\S.*)$")
PR_NUMBER_RE = re.compile(r"^(?P<description>.*)\s\(#(?P<pr_number>\d+)\)$")
VERSION_LINE_RE = re.compile(r'^version = "[^"]*"$', re.MULTILINE)
RELEASE_COMMIT_RE = re.compile(r"^chore: release v\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class ParsedCommit:
    type: str | None
    description: str
    pr_number: int | None


def parse_commit_subject(subject: str) -> ParsedCommit:
    """Split a commit subject into its conventional-commit parts.

    A subject that does not match the conventional-commit shape (e.g. the
    repo's very first commit) keeps its full text as the description, with
    no type or PR number.
    """
    match = COMMIT_RE.match(subject)
    if match is None:
        description, pr_number = _split_pr_number(subject)
        return ParsedCommit(type=None, description=description, pr_number=pr_number)
    description, pr_number = _split_pr_number(match.group("description"))
    return ParsedCommit(
        type=match.group("type"),
        description=description,
        pr_number=pr_number,
    )


def _split_pr_number(description: str) -> tuple[str, int | None]:
    match = PR_NUMBER_RE.match(description)
    if match is None:
        return description, None
    return match.group("description"), int(match.group("pr_number"))


def is_release_commit(subject: str) -> bool:
    """Return whether a subject is an exact release commit."""
    return RELEASE_COMMIT_RE.match(subject) is not None


def group_commits(commits: list[ParsedCommit]) -> dict[str, list[ParsedCommit]]:
    """Group commits into the fixed section order, omitting empty sections."""
    buckets: dict[str, list[ParsedCommit]] = {name: [] for name in SECTION_ORDER}
    for commit in commits:
        section = TYPE_SECTIONS.get(commit.type, "Other")
        buckets[section].append(commit)
    return {name: entries for name, entries in buckets.items() if entries}


def render_version_section(version: str, release_date: str, commits: list[ParsedCommit]) -> str:
    """Render one version's markdown section: a header plus its grouped entries."""
    lines = [f"## [{version}] - {release_date}"]
    for section_name, entries in group_commits(commits).items():
        lines.append("")
        lines.append(f"### {section_name}")
        for entry in entries:
            suffix = f" (#{entry.pr_number})" if entry.pr_number is not None else ""
            lines.append(f"- {entry.description}{suffix}")
    return "\n".join(lines)


def bump_version(version: str, level: str) -> str:
    """Raise a MAJOR.MINOR.PATCH string by one major, minor, or patch step."""
    if level not in BUMP_LEVELS:
        raise ValueError(f"unknown bump level {level!r}; expected major, minor, patch")
    major, minor, patch = (int(part) for part in version.split("."))
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def bump_version_line(pyproject_text: str, new_version: str) -> str:
    """Replace the project version line without changing other text."""
    new_text, count = VERSION_LINE_RE.subn(f'version = "{new_version}"', pyproject_text, count=1)
    if count != 1:
        raise RuntimeError("pyproject.toml has no top-level version line")
    return new_text


# ---- git effects (not unit tested; exercised against the real repo) ---------


def _run(*args: str) -> str:
    result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return result.stdout


def is_working_tree_dirty() -> bool:
    return bool(_run("git", "status", "--porcelain").strip())


def current_branch() -> str:
    return _run("git", "branch", "--show-current").strip()


def tag_exists(tag: str) -> bool:
    return bool(_run("git", "tag", "-l", tag).strip())


def list_tags() -> list[str]:
    """Return v-prefixed tags, oldest to newest by semver."""
    tags = [line for line in _run("git", "tag", "-l", "v*").splitlines() if line]
    return sorted(tags, key=lambda tag: tuple(int(part) for part in tag[1:].split(".")))


def commit_subjects(rev_range: str) -> list[str]:
    return [
        line
        for line in _run("git", "log", "--no-merges", "--pretty=%s", rev_range).splitlines()
        if line
    ]


def tag_date(tag: str) -> str:
    return _run("git", "log", "-1", "--format=%as", tag).strip()


def read_version() -> str:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def _parsed_commits(rev_range: str) -> list[ParsedCommit]:
    """Parse commits in a range while excluding release commits."""
    return [
        parse_commit_subject(subject)
        for subject in commit_subjects(rev_range)
        if not is_release_commit(subject)
    ]


def build_changelog(new_version: str) -> str:
    """Regenerate the full changelog from every existing tag plus history since."""
    tags = list_tags()
    sections: list[tuple[str, str, list[ParsedCommit]]] = []
    previous_tag = None
    for tag in tags:
        rev_range = f"{previous_tag}..{tag}" if previous_tag else tag
        sections.append((tag[1:], tag_date(tag), _parsed_commits(rev_range)))
        previous_tag = tag

    new_rev_range = f"{previous_tag}..HEAD" if previous_tag else "HEAD"
    sections.append((new_version, date.today().isoformat(), _parsed_commits(new_rev_range)))

    rendered = [render_version_section(*section) for section in reversed(sections)]
    return "# Changelog\n\n" + "\n\n".join(rendered) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1 or (argv and argv[0] not in BUMP_LEVELS):
        print(f"usage: release.py [{'|'.join(BUMP_LEVELS)}]", file=sys.stderr)
        return 2
    bump = argv[0] if argv else None

    # The command-text guard cannot see git spawned from Python, so this path refuses itself.
    if os.environ.get("CLAUDECODE"):
        print(
            "refusing to release: an agent works in a worktree and lands via PR (AGENTS.md); "
            "a release is cut by the operator running this script directly",
            file=sys.stderr,
        )
        return 1

    if is_working_tree_dirty():
        print("refusing to release: working tree is dirty", file=sys.stderr)
        return 1
    branch = current_branch()
    if branch != "main":
        print(f"refusing to release: on branch {branch!r}, expected main", file=sys.stderr)
        return 1

    current_version = read_version()
    if bump is None:
        target_version = current_version
        if tag_exists(f"v{target_version}"):
            print(
                f"refusing to release: v{target_version} is already tagged; "
                "pass a bump level (major, minor, patch) to cut a new version",
                file=sys.stderr,
            )
            return 1
    else:
        target_version = bump_version(current_version, bump)
        if tag_exists(f"v{target_version}"):
            print(f"refusing to release: v{target_version} already exists", file=sys.stderr)
            return 1
        PYPROJECT.write_text(
            bump_version_line(PYPROJECT.read_text(encoding="utf-8"), target_version),
            encoding="utf-8",
        )

    CHANGELOG.write_text(build_changelog(target_version), encoding="utf-8")

    # Refresh unconditionally: even on the no-bump path this is a no-op that stages nothing new.
    _run("uv", "lock")
    _run(
        "git",
        "add",
        str(PYPROJECT.relative_to(REPO_ROOT)),
        str(CHANGELOG.relative_to(REPO_ROOT)),
        str(LOCKFILE.relative_to(REPO_ROOT)),
    )
    _run("git", "commit", "-m", f"chore: release v{target_version}")
    # Annotated: the printed push carries annotated tags only.
    _run("git", "tag", "-a", f"v{target_version}", "-m", f"v{target_version}")

    print(f"Released v{target_version}.")
    print("Run: git push --follow-tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
