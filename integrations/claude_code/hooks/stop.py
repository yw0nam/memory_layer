"""Claude Code Stop hook: append the assistant's final answer to the turn lane.

Reads the hook payload on stdin and posts `last_assistant_message` with role
`assistant`. Never blocks: every failure is fail-open, exit 0. Stdlib only.
"""

from __future__ import annotations

import json
import sys

from hook_common import Config
from hook_common import emit
from hook_common import post_turn


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        cfg = Config.from_env()
        message = payload.get("last_assistant_message")
        if not cfg.configured or not message:
            return 0
        notice = post_turn(
            cfg,
            payload.get("session_id") or "unknown",
            "assistant",
            message,
            payload.get("cwd") or "",
        )
        emit("Stop", None, notice)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
