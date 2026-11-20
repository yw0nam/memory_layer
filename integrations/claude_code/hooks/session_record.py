"""Claude Code Stop, SessionEnd and SessionStart hook: start the session record uploader.

Stop and SessionEnd write the session's pending checkpoint, then spawn the uploader for its
transcript; SessionStart spawns the sweep. The uploader runs detached and the hook exits at
once. Prints nothing and always exits 0. Stdlib only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hook_common import Config
from session_record import checkpoint

HOOKS = Path(__file__).resolve().parent


def _spawn(argv: list[str]) -> None:
    subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd=HOOKS,
    )


def run(payload: dict, spawn=_spawn) -> None:
    if not Config.from_env().configured:
        return
    command = [sys.executable, "-m", "session_record.uploader"]
    if payload.get("hook_event_name") == "SessionStart":
        spawn([*command, "--sweep"])
        return
    path, session, cwd = payload["transcript_path"], payload["session_id"], payload["cwd"]
    checkpoint.pending(path, session, cwd)
    spawn([*command, path, session, cwd])


def main() -> int:
    try:
        run(json.load(sys.stdin))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
