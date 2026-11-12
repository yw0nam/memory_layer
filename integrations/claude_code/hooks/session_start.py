"""Claude Code SessionStart hook: deliver this repo's handoff headers.

Prints one <memory-context> fence holding the handoffs, or nothing when there are none, and records the
connection state to `status.json` in the plugin data directory. The hook never claims a handoff. Config comes from the plugin
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
HANDOFF_FETCH_LIMIT = 100
HANDOFF_HEADER = (
    "Memory: pending handoffs for this repo. Nothing is claimed; ask to continue one to claim it."
)
COMPLETED_LINE = (
    "Memory: completed handoffs delivered for this repo: {count}. "
    "Claim one to read it; list_messages names their ids."
)


def format_handoffs(rows: list[dict]) -> str:
    """The first open handoffs under the header, then one line counting the completed ones."""
    open_lines, completed = [], 0
    for row in rows:
        if row.get("status") == "completed":
            completed += 1
            continue
        if len(open_lines) == HANDOFF_LIMIT:
            continue
        subject = defuse(" ".join(str(row["subject"]).split()))
        created = str(row.get("created_at", ""))[:10]
        open_lines.append(
            f"- {created}  {subject}  (status: {row.get('status', '?')}, id: {row['id']})"
        )
    blocks = []
    if open_lines:
        blocks.append("\n".join([HANDOFF_HEADER, *open_lines]))
    if completed:
        blocks.append(COMPLETED_LINE.format(count=completed))
    return "\n".join(blocks)


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
            {"purpose": "handoff", "scope": scope, "limit": str(HANDOFF_FETCH_LIMIT)},
        )
        return format_handoffs(rows)
    except Exception:
        return ""


def run_hook(payload: dict, get, remote_of=git_origin) -> str:
    """This repo's handoffs in one fence via `get`, or "" when there are none."""
    handoffs = _handoffs(payload, get, remote_of)
    if not handoffs:
        return ""
    return "\n".join(["<memory-context>", handoffs, "</memory-context>"])


def _write_status(cfg: Config, reachable: bool, error: str | None) -> None:
    status = {
        "configured": cfg.configured,
        "url": cfg.url,
        "author": cfg.author,
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

        emit(run_hook(payload, fetch), None)
        try:
            _write_status(cfg, not failure, failure[0] if failure else None)
        except Exception:
            pass
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
