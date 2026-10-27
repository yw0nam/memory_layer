"""Claude Code UserPromptSubmit hook: capture the prompt as a `user` turn.

Reads the hook payload on stdin and appends the prompt to the turn lane;
prompts starting with `/` or `!` are not captured. Every failure mode is
fail-open: exit 0, never a blocked prompt. Stdlib only.

Config comes from the plugin options (`CLAUDE_PLUGIN_OPTION_*`).
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
        if not cfg.configured:
            return 0
        prompt = payload.get("prompt") or ""
        if not prompt.strip() or prompt.lstrip().startswith(("/", "!")):
            return 0
        notice = post_turn(
            cfg,
            payload.get("session_id") or "unknown",
            "user",
            prompt,
            payload.get("cwd") or "",
        )
        emit(None, notice)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
