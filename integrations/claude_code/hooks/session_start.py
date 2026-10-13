"""Claude Code SessionStart hook: deliver this agent's profile and this repo's pending handoffs.

Prints one <memory-context> fence and records the connection state to `status.json` in
the plugin data directory. The hook never claims a handoff. Config comes from the plugin
options (`CLAUDE_PLUGIN_OPTION_*`). Every failure mode is fail-open: exit 0. Stdlib only.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from datetime import timezone

from hook_common import Config
from hook_common import data_dir
from hook_common import defuse
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
    return defuse("\n".join(lines))


def format_handoffs(rows: list[dict]) -> str:
    """The handoff header and one line per handoff; empty for none."""
    lines = []
    for row in rows[:HANDOFF_LIMIT]:
        subject = defuse(" ".join(str(row["subject"]).split()))
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


def run_hook(payload: dict, get, remote_of=git_origin, *, owner: str) -> str:
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
                if path == "/profiles":
                    failure.append(str(exc))
                raise

        block = run_hook(payload, fetch, owner=cfg.author)
        emit("SessionStart", block, None)
        try:
            _write_status(cfg, not failure, failure[0] if failure else None)
        except Exception:
            pass
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
