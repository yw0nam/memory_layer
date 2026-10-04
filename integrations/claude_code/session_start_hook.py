"""Claude Code SessionStart hook: deliver this agent's profile and this repo's handoffs.

Reads the hook payload on stdin and prints one <memory-context> fence holding,
in order, the configured owner's profile (`GET /profiles?owner=<owner>`) and the
pending handoffs addressed to this repository. The profile block names the owner,
then prints `## user (v<n>)` and `## self (v<n>)`, each with its content or
`(empty)`, so the session always knows the user version a proposal must be
written against; when a proposal awaits the user's approval it adds a notice
naming the proposal id and the approval CLI, never the proposal's content. A
failed or malformed profile fetch prints no profile block and no version.
The handoff scope `repo:<host>/<path>` comes from the `origin` remote of the
payload's `cwd`; a cwd outside a git repository, without an `origin` remote, or
whose origin is not a remote host gets the profile alone. At most ten handoffs
print, newest first; the hook never claims one: the session claims a handoff
only when asked to continue it. The profile and the handoffs are fetched
independently, so a failure of either keeps the other. Every failure mode is
fail-open: no output, exit 0. Stdlib only — the script runs under whatever
python3 Claude Code invokes, outside any venv.

The hook runs at every SessionStart source (startup, resume, clear, compact)
through one settings.json entry without a matcher, so the profile returns after
a compaction. Its two requests take up to 3 seconds each, so the entry's
timeout is 10 seconds. Installation, the settings.json entry, and the key file
are shared with the UserPromptSubmit hook and documented in prefetch_hook.py.

Config via environment, every var optional: MEMORY_BASE_URL (default
http://127.0.0.1:8010), MEMORY_BASE_API_KEY, MEMORY_BASE_ENV (env file
holding the key; default ~/.config/memory-base/env), MEMORY_BASE_AUTHOR (the
profile owner, an author slug of the key; default claude-code).
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
DEFAULT_OWNER = "claude-code"
PROFILE_HEADER = "Memory: standing profile for {owner}. Apply it to every task."
PROFILE_CLI = "~/.config/memory-base/mb_profile.py"
PENDING_NOTICE = (
    "A proposed change to the user profile (proposal {id}) awaits the user's approval. "
    f"Ask the user to run `! python3 {PROFILE_CLI} show {{id}}` to inspect its diff, then "
    "approve or reject it with the memory-profile-approval skill."
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


def _defuse(text: str) -> str:
    return _FENCE_TAG.sub("[memory-context]", text)


def _part(profile: dict, name: str) -> tuple[int, str]:
    version, body = profile[f"{name}_version"], profile[name]
    if type(version) is not int:
        raise ValueError(f"{name}_version is not an integer")
    if body is None:
        return version, ""
    if not isinstance(body, dict) or not isinstance(body.get("content"), str):
        raise ValueError(f"{name} is malformed")
    return version, body["content"]


def format_profile(owner: str, profile: dict) -> str:
    """The owner's header, both parts under their version lines, and a pending notice.

    Raises on a malformed profile, so a bad response never prints a fabricated version.
    """
    if not isinstance(profile, dict):
        raise ValueError("profile is not an object")
    lines = [PROFILE_HEADER.format(owner=owner)]
    for name in ("user", "self"):
        version, content = _part(profile, name)
        lines += [f"## {name} (v{version})", content or "(empty)"]
    pending = profile["pending_proposal"]
    if pending is not None:
        if not isinstance(pending, dict) or type(pending.get("id")) is not int:
            raise ValueError("pending_proposal is malformed")
        lines.append(PENDING_NOTICE.format(id=pending["id"]))
    return _defuse("\n".join(lines))


def format_handoffs(rows: list[dict]) -> str:
    """The handoff header and one line per handoff; empty for none."""
    lines = []
    for row in rows[:HANDOFF_LIMIT]:
        subject = _defuse(" ".join(str(row["subject"]).split()))
        created = str(row.get("created_at", ""))[:10]
        lines.append(f"- {created}  {subject}  (status: {row.get('status', '?')}, id: {row['id']})")
    return "\n".join([HANDOFF_HEADER, *lines]) if lines else ""


def _profile(get, owner: str) -> str:
    try:
        return format_profile(owner, get("/profiles", {"owner": owner}))
    except Exception:
        return ""


def _handoffs(payload: dict, get, remote_of) -> str:
    cwd = payload.get("cwd") or ""
    if not cwd:
        return ""
    try:
        scope = repo_scope(remote_of(cwd))
        if scope is None:
            return ""
        rows = get(
            "/messages",
            {"purpose": "handoff", "scope": scope, "limit": str(HANDOFF_LIMIT)},
        )
        return format_handoffs(rows)
    except Exception:
        return ""


def run_hook(payload: dict, get, remote_of=git_origin, owner: str = DEFAULT_OWNER) -> str:
    """Fetch the owner's profile and this repo's pending handoffs via `get`; one fence, or empty."""
    parts = [part for part in (_profile(get, owner), _handoffs(payload, get, remote_of)) if part]
    if not parts:
        return ""
    return "\n".join(["<memory-context>", "\n\n".join(parts), "</memory-context>"])


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
    def get(path: str, params: dict):
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        req = urllib.request.Request(
            f"{url.rstrip('/')}{path}{query}",
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
        owner = os.environ.get("MEMORY_BASE_AUTHOR") or DEFAULT_OWNER
        block = run_hook(payload, _get_from_server(url, api_key), owner=owner)
        if block:
            print(block)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
