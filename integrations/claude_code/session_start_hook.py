"""Claude Code SessionStart hook: announce this repo's pending handoffs.

Reads the hook payload on stdin, derives the handoff scope
`repo:<host>/<path>` from the `origin` remote of the payload's `cwd`, lists the
pending handoffs addressed to that scope, and prints at most ten of them,
newest first, inside a <memory-context> fence. It never claims one: the session
claims a handoff only when asked to continue it. A cwd outside a git
repository, without an `origin` remote, or whose origin is not a remote host
prints nothing. Every failure mode is fail-open: no output, exit 0. Stdlib
only — the script runs under whatever python3 Claude Code invokes, outside any
venv.

The listing sends no namespace filter, so it spans every namespace the key
allows. Installation, the settings.json entry, and the key file are shared with
the UserPromptSubmit hook and documented in prefetch_hook.py.

Config via environment, every var optional: MEMORY_BASE_URL (default
http://127.0.0.1:8010), MEMORY_BASE_API_KEY, MEMORY_BASE_ENV (env file
holding the key; default ~/.config/memory-base/env).
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HANDOFF_LIMIT = 10
HTTP_TIMEOUT_SECONDS = 3.0
GIT_TIMEOUT_SECONDS = 1.0
HANDOFF_HEADER = (
    "Memory: pending handoffs for this repo. Nothing is claimed; ask to continue one to claim it."
)

_FENCE_TAG = re.compile(r"<\s*/?\s*memory-context", re.IGNORECASE)
_SCP_ORIGIN = re.compile(r"(?:[^:@/]+@)?(?P<host>[^:/]+):(?P<path>.+)")
_ORIGIN_HOST = re.compile(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def _is_local_host(host: str) -> bool:
    host = host.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def repo_scope(origin: str | None) -> str | None:
    """The server's repo scope for a git origin, or None when it is not portable.

    Matches `normalize_scope` on the server: the hostname is lowercased, path
    case is kept, and a trailing `.git` and slashes are dropped. Credentials,
    user names, and ports never reach the scope.
    """
    origin = (origin or "").strip()
    if not origin or any(char.isspace() for char in origin):
        return None
    if "://" in origin:
        parts = urllib.parse.urlsplit(origin)
        host, path = parts.hostname or "", parts.path
    elif (scp := _SCP_ORIGIN.fullmatch(origin)) is not None:
        host, path = scp["host"], scp["path"]
    else:
        return None
    path = path.split("?", 1)[0].split("#", 1)[0].strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    if not path or not _ORIGIN_HOST.fullmatch(host) or _is_local_host(host):
        return None
    return f"repo:{host.lower()}/{path}"


def git_origin(cwd: str) -> str | None:
    """The `origin` remote URL of the repository holding cwd, or None."""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except Exception:
        return None
    url = result.stdout.strip()
    return url if result.returncode == 0 and url else None


def format_handoffs(rows: list[dict]) -> str:
    """One line per handoff inside a memory-context fence; empty for none."""
    lines = []
    for row in rows[:HANDOFF_LIMIT]:
        subject = _FENCE_TAG.sub("[memory-context]", " ".join(str(row["subject"]).split()))
        created = str(row.get("created_at", ""))[:10]
        lines.append(f"- {created}  {subject}  (status: {row.get('status', '?')}, id: {row['id']})")
    if not lines:
        return ""
    return "\n".join(["<memory-context>", HANDOFF_HEADER, *lines, "</memory-context>"])


def run_hook(payload: dict, get, remote_of=git_origin) -> str:
    """Derive the scope from cwd, list its pending handoffs via `get`, and format them."""
    cwd = payload.get("cwd") or ""
    if not cwd:
        return ""
    scope = repo_scope(remote_of(cwd))
    if scope is None:
        return ""
    try:
        rows = get(
            "/messages",
            {"purpose": "handoff", "scope": scope, "limit": str(HANDOFF_LIMIT)},
        )
        return format_handoffs(rows)
    except Exception:
        return ""


def _resolve_api_key() -> str:
    key = os.environ.get("MEMORY_BASE_API_KEY", "")
    if key:
        return key
    env_file = os.environ.get("MEMORY_BASE_ENV", "") or str(
        Path.home() / ".config" / "memory-base" / "env"
    )
    try:
        for line in Path(env_file).read_text().splitlines():
            if line.startswith("MEMORY_BASE_API_KEY="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


def _get_from_server(url: str, api_key: str):
    def get(path: str, params: dict) -> list[dict]:
        req = urllib.request.Request(
            f"{url.rstrip('/')}{path}?{urllib.parse.urlencode(params)}",
            headers={"X-API-Key": api_key},
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read())

    return get


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        api_key = _resolve_api_key()
        if not api_key:
            return 0
        url = os.environ.get("MEMORY_BASE_URL", "http://127.0.0.1:8010")
        block = run_hook(payload, _get_from_server(url, api_key))
        if block:
            print(block)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
