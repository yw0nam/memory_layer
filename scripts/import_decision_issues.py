#!/usr/bin/env python3
"""Import the closed `decision` issues of one repository into the decision lane.

One-time and resumable: an issue whose URL is already in some decision's links
is skipped, so the run can be repeated until it completes. gh reads the issues,
the memory-base REST API records them. A dry run prints what it would record
and writes nothing.

Usage: import_decision_issues.py --repo OWNER/NAME --url MEMORY_BASE_URL [--dry-run]

Environment: MEMORY_BASE_API_KEY, MEMORY_BASE_AUTHOR, CLAUDE_PLUGIN_AUTHOR.
The two author values must be set and equal; the plugin's decisions must not
land under another persona.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request

GH_TIMEOUT_S = 30
REST_TIMEOUT_S = 30
ISSUE_LIST_LIMIT = 1000
LIST_LIMIT = 200
TITLE_MAX_CHARS = 200
BODY_MAX_CHARS = 1500
DECISION_HEADING = re.compile(r"^#{1,6}\s*Decision\b", re.IGNORECASE)


class GhError(Exception):
    """A gh call failed; the run stops before writing anything."""


class RequestFailed(Exception):
    """A memory-base REST call failed; the run stops."""


def run_gh(args: list[str]) -> str:
    """gh's stdout, or GhError naming the failed call."""
    try:
        result = subprocess.run(
            ["gh", *args],
            env={**os.environ, "GH_PROMPT_DISABLED": "1"},
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise GhError(f"gh {' '.join(args)} timed out") from exc
    if result.returncode != 0:
        raise GhError(f"gh {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def list_issues(repo: str) -> list[dict]:
    """The repository's closed `decision` issues, ascending issue number."""
    raw = run_gh(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--label",
            "decision",
            "--state",
            "closed",
            "--limit",
            str(ISSUE_LIST_LIMIT),
            "--json",
            "number,title,closedAt,url",
        ]
    )
    try:
        issues = sorted(json.loads(raw), key=lambda issue: issue["number"])
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        raise GhError(f"unexpected gh issue list output: {exc}") from exc
    return issues


def issue_comments(repo: str, number: int) -> list[dict]:
    """All comments of one issue, the --slurp pages flattened."""
    raw = run_gh(["api", "--paginate", "--slurp", f"repos/{repo}/issues/{number}/comments"])
    try:
        pages = json.loads(raw)
        return [comment for page in pages for comment in page]
    except (json.JSONDecodeError, TypeError) as exc:
        raise GhError(f"unexpected gh comments output for #{number}: {exc}") from exc


def closing_text(comments: list[dict], issue_url: str) -> str:
    """The closing decision text of an issue, cut to BODY_MAX_CHARS with a pointer suffix.

    The last comment whose body has a `Decision` heading wins; otherwise all
    comment bodies joined by blank lines.
    """
    bodies = [str(comment.get("body") or "") for comment in comments]
    text = ""
    for body in reversed(bodies):
        if any(DECISION_HEADING.match(line) for line in body.splitlines()):
            text = body.strip()
            break
    if not text:
        text = "\n\n".join(body.strip() for body in bodies)
    if len(text) <= BODY_MAX_CHARS:
        return text
    suffix = f" [cut; full text: {issue_url}]"
    return text[: BODY_MAX_CHARS - len(suffix)].rstrip() + suffix


def rest(url: str, *, api_key: str, author: str, payload: dict | None = None) -> dict:
    """The parsed JSON response of one memory-base call, with the plugin hooks' headers."""
    headers = {"X-API-Key": api_key, "X-Author": author}
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=REST_TIMEOUT_S) as response:
            return json.loads(response.read())
    except (OSError, json.JSONDecodeError) as exc:
        raise RequestFailed(str(exc)) from exc


def imported_links(base: str, scope: str, *, api_key: str, author: str) -> set[str]:
    """Every link in the scope's decisions, active or not; the resume check."""
    query = urllib.parse.urlencode({"scope": scope, "status": "all", "limit": LIST_LIMIT})
    rows = rest(f"{base}/decisions?{query}", api_key=api_key, author=author)
    links: set[str] = set()
    for row in rows:
        query = urllib.parse.urlencode({"scope": scope})
        detail = rest(f"{base}/decisions/{row['number']}?{query}", api_key=api_key, author=author)
        links.update(detail["links"])
    return links


def parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, metavar="OWNER/NAME", help="GitHub repository")
    parser.add_argument("--url", required=True, help="memory-base REST base URL")
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would be recorded; write nothing"
    )
    args = parser.parse_args(argv)
    if re.fullmatch(r"[^/\s]+/[^/\s]+", args.repo) is None:
        parser.error("--repo must be OWNER/NAME")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    author = os.environ.get("MEMORY_BASE_AUTHOR")
    if not author or author != os.environ.get("CLAUDE_PLUGIN_AUTHOR"):
        print(
            "error: MEMORY_BASE_AUTHOR and CLAUDE_PLUGIN_AUTHOR must be set and equal",
            file=sys.stderr,
        )
        return 2
    api_key = os.environ.get("MEMORY_BASE_API_KEY")
    if not api_key:
        print("error: MEMORY_BASE_API_KEY is not set", file=sys.stderr)
        return 2
    print(f"Author: {author}")
    base = args.url.rstrip("/")
    scope = f"repo:github.com/{args.repo}"
    try:
        already = imported_links(base, scope, api_key=api_key, author=author)
        issues = list_issues(args.repo)
    except (GhError, RequestFailed) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    recorded = skipped = 0
    for issue in issues:
        url = issue["url"]
        if url in already:
            print(f"#{issue['number']} skipped (already imported)")
            skipped += 1
            continue
        try:
            comments = issue_comments(args.repo, issue["number"])
        except GhError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        text = closing_text(comments, url)
        if not text:
            print(f"error: #{issue['number']} has no closing decision text", file=sys.stderr)
            return 1
        record = {
            "scope": scope,
            "title": issue["title"][:TITLE_MAX_CHARS],
            "decision": text,
            "links": [url],
        }
        if issue.get("closedAt"):
            record["decided_on"] = issue["closedAt"][:10]
        if args.dry_run:
            print(f"#{issue['number']} -> D-? (dry run) {json.dumps(record, ensure_ascii=False)}")
            recorded += 1
            continue
        try:
            result = rest(f"{base}/decisions", api_key=api_key, author=author, payload=record)
        except RequestFailed as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"#{issue['number']} -> D-{result['number']}")
        recorded += 1
    verb = "would record" if args.dry_run else "recorded"
    print(f"{verb} {recorded}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
