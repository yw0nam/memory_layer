"""Claude Code SessionStart hook: deliver this agent's profile and this repo's handoffs.

Reads the hook payload on stdin and prints one <memory-context> fence holding,
in order, the configured author's profile (`GET /profiles?owner=<author>`) and
the pending handoffs addressed to this repository. The profile block names the
owner, then the session id line, then prints `## user (v<n>)` and
`## self (v<n>)`, each with its content or `(empty)`, so the session always
knows the user version an update must be written against. A failed or
malformed profile fetch prints no profile block and no version.
The handoff scope `repo:<host>/<path>` comes from the `origin` remote of the
payload's `cwd`; a cwd outside a git repository, without an `origin` remote, or
whose origin is not a remote host gets the profile alone. At most ten handoffs
print, newest first; the hook never claims one: the session claims a handoff
only when asked to continue it. The profile and the handoffs are fetched
independently, so a failure of either keeps the other. Every failure mode is
fail-open: no output, exit 0. Stdlib only.

The hook runs at every SessionStart source (startup, resume, clear, compact),
so the profile returns after a compaction. Each run records the connection
state to `status.json` in the plugin data directory. Config comes from the
plugin options (`CLAUDE_PLUGIN_OPTION_*`).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from datetime import timezone

from hook_common import DEFAULT_AUTHOR
from hook_common import Config
from hook_common import data_dir
from hook_common import emit
from hook_common import get
from hook_common import git_origin
from hook_common import repo_scope

HANDOFF_LIMIT = 10
HANDOFF_HEADER = (
    "Memory: pending handoffs for this repo. Nothing is claimed; ask to continue one to claim it."
)
PROFILE_HEADER = "Memory: standing profile for {owner}. Apply it to every task."
SESSION_LINE = (
    "Session: claude_code {session_id} "
    "(pass it as session_id with origin claude_code when a note is distilled from this session)"
)

_FENCE_TAG = re.compile(r"<\s*/?\s*memory-context", re.IGNORECASE)


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


def format_profile(owner: str, session_id: str, profile: dict) -> str:
    """The owner's header, and both parts under their version lines.

    Raises on a malformed profile, so a bad response never prints a fabricated version.
    """
    if not isinstance(profile, dict):
        raise ValueError("profile is not an object")
    lines = [PROFILE_HEADER.format(owner=owner), SESSION_LINE.format(session_id=session_id)]
    for name in ("user", "self"):
        version, content = _part(profile, name)
        lines += [f"## {name} (v{version})", content or "(empty)"]
    return _defuse("\n".join(lines))


def format_handoffs(rows: list[dict]) -> str:
    """The handoff header and one line per handoff; empty for none."""
    lines = []
    for row in rows[:HANDOFF_LIMIT]:
        subject = _defuse(" ".join(str(row["subject"]).split()))
        created = str(row.get("created_at", ""))[:10]
        lines.append(f"- {created}  {subject}  (status: {row.get('status', '?')}, id: {row['id']})")
    return "\n".join([HANDOFF_HEADER, *lines]) if lines else ""


def _profile(get, owner: str, session_id: str) -> str:
    try:
        return format_profile(owner, session_id, get("/profiles", {"owner": owner}))
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


def run_hook(payload: dict, get, remote_of=git_origin, owner: str = DEFAULT_AUTHOR) -> str:
    """Fetch the owner's profile and this repo's pending handoffs via `get`; one fence, or empty."""
    session_id = payload.get("session_id") or "unknown"
    parts = [
        part
        for part in (_profile(get, owner, session_id), _handoffs(payload, get, remote_of))
        if part
    ]
    if not parts:
        return ""
    return "\n".join(["<memory-context>", "\n\n".join(parts), "</memory-context>"])


def _write_status(cfg: Config, reachable: bool, error: str | None) -> None:
    status = {
        "configured": cfg.configured,
        "url": cfg.url,
        "author": cfg.author,
        "namespace": cfg.namespace,
        "reachable": reachable,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": error,
    }
    (data_dir() / "status.json").write_text(json.dumps(status, indent=2))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        cfg = Config.from_env()
        if not cfg.configured:
            _write_status(cfg, False, None)
            return 0
        failure: list[str] = []

        def fetch(path: str, params: dict):
            try:
                return get(cfg, path, params)
            except Exception as exc:
                failure.append(str(exc))
                raise

        block = run_hook(payload, fetch, owner=cfg.author)
        _write_status(cfg, not failure, failure[0] if failure else None)
        emit("SessionStart", block, None)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
